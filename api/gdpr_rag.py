"""GDPR RAG engine: hybrid search -> reranker -> quote-verified answer (Groq)."""
import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",") if m.strip()]
N_CANDIDATES = int(os.getenv("N_CANDIDATES", "20"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "1024"))   # tested value; lower = faster but must be re-checked
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")              # tested value; "low" = faster but must be re-checked
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))   # thinking tokens count too; 2500 could cut the JSON short
DAILY_LLM_CALLS = int(os.getenv("DAILY_LLM_CALLS", "60"))
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
NOT_FOUND = "Not found in the GDPR text."

STOP = set("the and for that with this from which shall have been will not are was were their there where when than into such other under upon".split())
FOLLOW = re.compile(r"\b(it|its|they|them|their|that|this|those|these|what about|and if|also|and for)\b", re.I)


# ----------------------------------------------------------------- helpers
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP}


def grams(t, n=4):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def support(quote, text):
    """Share of the quote's 4-word sequences that really appear in the text."""
    qg = grams(tok(quote))
    return len(qg & grams(tok(text))) / len(qg) if qg else 0.0


def cite(c):
    pg = f"p. {c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")
    if c["type"] == "article":
        para = f"({c['paragraph']})" if c.get("paragraph") else ""
        return f"Art. {c['article_no']}{para}, {pg}"
    return f"Recital {c['recital_no']}, {pg}"


def parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def _num_excerpt(answer, text, width=350):
    norm = lambda s: s.replace("\u202f", " ").replace("\u00a0", " ").lower()
    tk = lambda s: set(re.findall(r"[a-z]{3,}|\d+", norm(s)))
    aw = tk(answer)
    nums = {t for t in aw if t.isdigit()}
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+|\n", text) if len(p.strip()) > 20]
    if not parts:
        return text[:width]
    score = lambda i: (len(aw & tk(parts[i])) + 3 * len(nums & tk(parts[i]))) / (len(tk(parts[i])) ** 0.5 + 1)
    best = max(range(len(parts)), key=score)
    out, j = parts[best], best + 1
    while len(out) < width and j < len(parts):
        out += " " + parts[j]
        j += 1
    return out[:width] + ("…" if len(out) > width else "")


def excerpt(quote, text, max_chars=500):
    """The full clause(s) of the passage that contain the verified quote."""
    qt = set(tok(quote))
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+", text.replace("\n", " ")) if p.strip()]
    keep = []
    for p in parts:
        pt = set(tok(p))
        if pt and (len(pt & qt) / len(pt) >= 0.7 or len(pt & qt) / max(1, len(qt)) >= 0.6):
            if p not in keep:
                keep.append(p)
    out = " ".join(keep) if keep else _num_excerpt(quote, text, 400)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


def load_prompt():
    p = (DATA_DIR / "prompt.txt").read_text(encoding="utf-8")
    anchor = "- Never use outside knowledge."
    extra = []
    if "listed case" not in p:
        extra.append("- If a passage lists cases (a), (b), (c) after a lead-in such as 'in any case where', write one claim per listed case, each with its own quote. Never use the lead-in alone as a claim.")
    if "who or what it applies to" not in p:
        extra.append("- Begin each claim by saying who or what it applies to (for example 'the controller', 'a supervisory authority member'). If the passages only concern a different subject or situation than the one asked about, return {\"claims\": []}.")
    if extra and anchor in p:
        p = p.replace(anchor, "\n".join(extra) + "\n" + anchor)
    return p


def _res(status, answer, **kw):
    base = {"status": status, "answer": answer, "claims": [], "sources": [], "removed": 0, "model": None, "timings": {}}
    base.update(kw)
    return base


def _done(res):
    return {"type": "result", "result": res}


# ----------------------------------------------------------------- engine
class GDPRBot:
    def __init__(self):
        import torch
        from groq import Groq
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
        self.groq = Groq(api_key=key)

        s = json.loads((DATA_DIR / "settings.json").read_text(encoding="utf-8"))
        self.bonus = float(s.get("ARTICLE_BONUS", 0.10))
        self.min_score = float(s.get("MIN_SCORE", 0.015))
        self.min_support = float(s.get("MIN_SUPPORT", 0.8))
        self.min_overlap = float(s.get("MIN_OVERLAP", 0.3))

        self.chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.emb = np.load(DATA_DIR / "embeddings.npy")
        if len(self.chunks) != len(self.emb):
            raise RuntimeError(f"chunks ({len(self.chunks)}) and embeddings ({len(self.emb)}) do not match")
        self.prompt = load_prompt()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.embedder.max_seq_length = 1024
        self.reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=RERANK_MAX_LEN)

        self.qdrant = QdrantClient(":memory:")
        self.col = "gdpr"
        self.qdrant.create_collection(self.col, vectors_config=qm.VectorParams(size=self.emb.shape[1], distance=qm.Distance.COSINE))
        self.qdrant.upsert(self.col, points=[qm.PointStruct(id=i, vector=self.emb[i].tolist(), payload={"i": i})
                                             for i in range(len(self.chunks))])
        self.bm25 = BM25Okapi([tok(c["embed_text"]) for c in self.chunks])

        self.art_index = defaultdict(list)
        for i, c in enumerate(self.chunks):
            if c["type"] == "article":
                self.art_index[c["article_no"]].append(i)
        self.pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

        self.cache = OrderedDict()
        self._day, self._calls = date.today(), 0

        if os.getenv("WARMUP", "1") == "1":              # one dummy search so the first real question is not slow
            t = time.time()
            try:
                self.retrieve("When must a controller notify a personal data breach?")
            except Exception as e:
                print(f"[warm-up] skipped: {e}", flush=True)
            else:
                print(f"[timing] warm-up: {time.time() - t:.1f}s", flush=True)

    # ------------------------------------------------------------ search
    def vector_search(self, q, k=20):
        qv = self.embedder.encode([q], normalize_embeddings=True)[0].tolist()
        return [p.id for p in self.qdrant.query_points(self.col, query=qv, limit=k).points]

    def keyword_search(self, q, k=20):
        return [int(i) for i in np.argsort(-self.bm25.get_scores(tok(q)))[:k]]

    def article_lookup(self, q):
        m = re.search(r"\barticle\s+(\d{1,3})\b", q.lower())
        if not m:
            return []
        n = int(m.group(1))
        return [i for i, c in enumerate(self.chunks) if c["type"] == "article" and c["article_no"] == n]

    def hybrid_search(self, q, k=10, w_article_boost=1.3):
        scores = {}
        for ranking in (self.vector_search(q), self.keyword_search(q)):
            for r, i in enumerate(ranking):
                scores[i] = scores.get(i, 0) + 1 / (60 + r)
        for i in self.article_lookup(q):
            scores[i] = scores.get(i, 0) + 0.1
        for i in scores:
            if self.chunks[i]["type"] == "article":
                scores[i] *= w_article_boost
        return [self.chunks[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]

    def retrieve(self, q, n_final=5):
        cands = self.hybrid_search(q, k=N_CANDIDATES)
        pairs = [(q, c["embed_text"]) for c in cands]
        sc = [float(s) for s in self.reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)]
        adj = [s + (self.bonus if c["type"] == "article" else 0.0) for c, s in zip(cands, sc)]
        order = sorted(range(len(cands)), key=lambda i: -adj[i])[:n_final]
        return [(cands[i], sc[i]) for i in order]          # ranked WITH the bonus, scores stay RAW

    def retrieve_expanded(self, q, n_final=5, n_articles=2, max_chunks=10, max_siblings=8):
        top = self.retrieve(q, n_final)
        chosen = {c["chunk_id"]: (c, s) for c, s in top}
        arts_done, best = [], {}
        for c, s in top:
            if c["type"] == "article":
                best[c["article_no"]] = max(best.get(c["article_no"], 0), s)
                if c["article_no"] not in arts_done:
                    arts_done.append(c["article_no"])
            if len(arts_done) == n_articles:
                break
        for n in arts_done:                                  # add the neighbouring paragraphs
            idxs = self.art_index[n]
            idset = set(idxs)
            if len(idxs) <= max_siblings:
                pick = idxs
            else:
                hp = [self.pos[c["chunk_id"]] for c, _ in top if c["type"] == "article" and c["article_no"] == n]
                pick = sorted({j for p in hp for j in (p - 1, p, p + 1) if j in idset})
            for j in pick:
                c = self.chunks[j]
                if c["chunk_id"] not in chosen and len(chosen) < max_chunks:
                    chosen[c["chunk_id"]] = (c, best.get(n, 0))
        out, used = [], set()
        for n in arts_done:                                  # best article first, paragraphs in order
            grp = sorted([v for v in chosen.values() if v[0]["type"] == "article" and v[0]["article_no"] == n],
                         key=lambda v: self.pos[v[0]["chunk_id"]])
            for c, _ in grp:
                out.append((c, best.get(n, 0)))
                used.add(c["chunk_id"])
        out += [(c, s) for c, s in top if c["chunk_id"] not in used]
        return out, max(s for _, s in out)

    # ------------------------------------------------------------ AI call
    def _budget_ok(self):
        today = date.today()
        if today != self._day:
            self._day, self._calls = today, 0
        if self._calls >= DAILY_LLM_CALLS:
            return False
        self._calls += 1
        return True

    def _llm(self, system, user, effort="medium"):
        last = None
        for m in LLM_MODELS:
            try:
                r = self.groq.chat.completions.create(
                    model=m, temperature=0, max_tokens=LLM_MAX_TOKENS, reasoning_effort=effort,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                text = re.sub(r"<think>.*?</think>", "", (r.choices[0].message.content or "").strip(), flags=re.S).strip()
                usage = getattr(r, "usage", None)
                print(f"[llm] model={m} effort={effort} finish={r.choices[0].finish_reason} "
                      f"completion_tokens={getattr(usage, 'completion_tokens', '?')} chars={len(text)}", flush=True)
                return text, m
            except Exception as e:
                last = e
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"[llm] {m} is at its limit, trying the next model", flush=True)
                    continue                                 # try the backup model
                raise
        raise last

    def standalone(self, msg, prev):
        """Turn a short follow-up ('what about children?') into a full question."""
        if not prev or not (len(msg.split()) <= 12 and FOLLOW.search(msg)) or not self._budget_ok():
            return msg
        try:
            out, _ = self._llm("Rewrite the follow-up as ONE complete standalone question about the GDPR, using the previous question for context. Output only the question.",
                               f"Previous question: {prev}\nFollow-up: {msg}", effort="low")
            out = out.strip().split("\n")[0]
            return out if 5 <= len(out) <= 300 else msg
        except Exception:
            return msg

    # ------------------------------------------------------------ answer
    def ask(self, question, previous_question=None):
        """Blocking version: returns only the final result."""
        res = None
        for ev in self.ask_events(question, previous_question):
            if ev["type"] == "result":
                res = ev["result"]
        return res

    def ask_events(self, question, previous_question=None):
        """Yields status events while working, then exactly one result event."""
        t0 = time.time()
        question = " ".join(question.split())
        if previous_question and len(question.split()) <= 12 and FOLLOW.search(question):
            yield {"type": "status", "text": "Understanding your follow-up question..."}
        q = self.standalone(question, previous_question)
        key = q.lower()
        if key in self.cache:
            res = dict(self.cache[key])
            res.update(cached=True, searched_as=q, timings={}, seconds=round(time.time() - t0, 2))
        else:
            res = None
            for ev in self._answer_gen(q):
                if ev["type"] == "result":
                    res = ev["result"]
                else:
                    yield ev
            if res is None:
                res = _res("warning", "Something went wrong. Please try again.")
            res.update(searched_as=q, cached=False, seconds=round(time.time() - t0, 2))
            if res["status"] in ("answered", "refused"):
                self.cache[key] = dict(res)
                while len(self.cache) > 300:
                    self.cache.popitem(last=False)
        print(f"[timing] total {res['seconds']}s | {res['status']}{' | cached' if res['cached'] else ''} | {q[:70]}", flush=True)
        yield _done(res)

    def _answer_gen(self, q):
        yield {"type": "status", "text": "Searching the GDPR text and ranking the best passages..."}
        t_search = time.time()
        hits, top = self.retrieve_expanded(q)
        t_search = time.time() - t_search
        print(f"[timing] search + rerank: {t_search:.1f}s (top score {top:.3f})", flush=True)
        if top < self.min_score:
            yield _done(_res("refused", NOT_FOUND, timings={"search": round(t_search, 1)}))
            return
        labels = ", ".join(dict.fromkeys(
            f"Art. {c['article_no']}" if c["type"] == "article" else f"Recital {c['recital_no']}" for c, _ in hits))
        yield {"type": "status", "text": f"Found relevant text in {labels} ({t_search:.1f}s)."}
        if not self._budget_ok():
            yield _done(_res("warning", "The demo has reached its daily AI limit. Please try again tomorrow."))
            return

        yield {"type": "status", "text": "Asking the AI to answer only from these passages..."}
        ctx = "\n\n".join(f"[{i + 1}] ({cite(c)})\n{c['text']}" for i, (c, _) in enumerate(hits))
        t_llm = time.time()
        data, model = None, None
        try:
            for attempt in range(2):                         # retry once if the form is unreadable
                effort = LLM_EFFORT if attempt == 0 else "low"
                raw, model = self._llm(self.prompt, f"Context passages:\n\n{ctx}\n\nQuestion: {q}", effort=effort)
                data = parse_json(raw)
                if isinstance(data, dict):
                    break
                print(f"[llm] unreadable reply (attempt {attempt + 1}, model {model}): {raw[:300]!r}", flush=True)
        except Exception as e:
            print(f"[timing] AI call failed after {time.time() - t_llm:.1f}s: {str(e)[:200]}", flush=True)
            busy = "429" in str(e) or "rate limit" in str(e).lower()
            yield _done(_res("warning", "The AI service is busy. Please try again in a minute." if busy
                             else "The AI service had a problem. Please try again."))
            return
        t_ai = time.time() - t_llm
        print(f"[timing] AI call: {t_ai:.1f}s (model {model})", flush=True)
        timings = {"search": round(t_search, 1), "ai": round(t_ai, 1)}
        if not isinstance(data, dict):
            yield _done(_res("warning", "The AI reply could not be read. Please try again.", model=model, timings=timings))
            return

        yield {"type": "status", "text": f"The AI answered ({t_ai:.1f}s). Checking every quote against the GDPR text..."}
        claims = data.get("claims") or []
        if not claims:
            yield _done(_res("refused", NOT_FOUND, model=model, timings=timings))
            return

        good, dropped = [], 0
        for cl in claims:
            try:
                n, claim, quote = int(cl["passage"]), str(cl["claim"]).strip(), str(cl["quote"]).strip()
            except Exception:
                dropped += 1
                continue
            homes = [i for i, (c, _) in enumerate(hits) if support(quote, c["text"]) >= self.min_support]
            if not homes or len(tok(quote)) < 6:
                dropped += 1                                  # quote is not in the real text: delete the claim
                continue
            main = (n - 1) if (1 <= n <= len(hits) and (n - 1) in homes) else homes[0]
            art = hits[main][0].get("article_no")
            homes = [main] + [i for i in homes if i != main and art is not None and hits[i][0].get("article_no") == art][:2]
            cs = [hits[i][0] for i in homes]
            cw = words(claim)
            overlap = len(cw & words(quote)) / max(1, len(cw))
            if overlap >= self.min_overlap:
                good.append((claim, cs, quote))
            else:                                             # genuine quote, weak wording: show the quote itself
                good.append((f'The text states: "{quote}"', cs[:1], quote))
        if not good:
            yield _done(_res("warning", "Could not verify the answer against the text. Please rephrase the question.",
                             removed=dropped, model=model, timings=timings))
            return

        claims_out = [{"text": cl, "citations": [cite(c) for c in cs]} for cl, cs, _ in good]
        lines = [f"{c['text']} " + "".join(f"[{x}]" for x in c["citations"]) for c in claims_out]
        answer = lines[0] if len(lines) == 1 else "\n".join("- " + l for l in lines)

        grouped, pages = {}, {}
        for _, cs, quote in good:
            for c in cs:
                lst = grouped.setdefault(cite(c), [])
                ex = excerpt(quote, c["text"])
                if ex not in lst:
                    lst.append(ex)
                pages[cite(c)] = (c["page_start"], c["page_end"])
        sources = [{"citation": k, "excerpt": " | ".join(v), "page_start": pages[k][0], "page_end": pages[k][1]}
                   for k, v in grouped.items()]
        yield _done(_res("answered", answer, claims=claims_out, sources=sources, removed=dropped,
                         model=model, timings=timings))

    def info(self):
        return {"status": "ok", "chunks": len(self.chunks), "device": self.device,
                "llm_models": LLM_MODELS, "daily_ai_calls_used": self._calls, "daily_ai_calls_limit": DAILY_LLM_CALLS}