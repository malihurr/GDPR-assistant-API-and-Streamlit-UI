import os
import re
import time

import requests

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
KEY = os.getenv("API_KEY", "")
HEAD = {"x-api-key": KEY} if KEY else {}
flat = lambda s: re.sub(r"[\s\u00a0\u202f]+", " ", s.lower())

# (question, expected status, facts that must appear, accepted articles)
TESTS = [
    ("How long do I have to notify the authority about a data breach?", "answered", ["72 hours"], [33]),
    ("When can a controller refuse a request to erase personal data?", "answered", [], [17]),
    ("When is consent valid?", "answered", [], [7, 4, 8]),
    ("When must a data protection officer be designated?", "answered", ["large scale"], [37]),
    ("What is the maximum fine for the most serious infringements?", "answered", ["20 000 000"], [83]),
    ("What is the capital of France?", "refused", [], []),
    ("What does the CCPA say about the right to opt out of the sale of personal information?", "refused", [], []),
]

h = requests.get(f"{API}/health", timeout=10)
print("health:", h.status_code, h.json() if h.ok else h.text)
all_ok = True
for q, status, facts, arts in TESTS:
    t0 = time.time()
    r = requests.post(f"{API}/ask", json={"question": q}, headers=HEAD, timeout=300)
    if not r.ok:
        print("❌ HTTP", r.status_code, q, r.text[:200])
        all_ok = False
        continue
    d = r.json()
    text = flat(d["answer"] + " " + " ".join(s["excerpt"] for s in d["sources"]))
    cited = {int(x) for c in d["claims"] for cit in c["citations"] for x in re.findall(r"Art\. (\d+)", cit)}
    checks = [d["status"] == status] + [flat(f) in text for f in facts] + ([bool(cited & set(arts))] if arts else [])
    ok = all(checks)
    all_ok &= ok
    print(f"{'✅' if ok else '❌'} {time.time() - t0:5.1f}s [{d['status']}] cited {sorted(cited)} | {q}")
    if not ok:
        print("    ", d["answer"][:300].replace("\n", " "))
    time.sleep(3)
print("\nALL PASSED" if all_ok else "\nSOME FAILED: paste this output to me")