import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="GDPR Assistant", page_icon="⚖️", layout="centered")

EXAMPLES = [
    "How long do I have to notify the authority about a data breach?",
    "When is consent valid?",
    "When can a controller refuse a request to erase personal data?",
    "When must a data protection officer be designated?",
    "What is the maximum fine for the most serious infringements?",
]


def call_api(question, previous):
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    r = requests.post(f"{API_URL}/ask", json={"question": question, "previous_question": previous},
                      headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()


def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def warn(msg):
    return {"status": "warning", "answer": msg, "sources": [], "removed": 0}


def render(res):
    status = res.get("status")
    if status == "answered":
        st.markdown(res["answer"])
        if res.get("removed"):
            st.caption(f"⚠️ {res['removed']} statement(s) could not be verified against the text and were removed.")
    elif status == "refused":
        st.info(res["answer"])
    else:
        st.warning(res["answer"])
    if res.get("sources"):
        with st.expander(f"Sources ({len(res['sources'])}) - exact text from the GDPR"):
            for s in res["sources"]:
                st.markdown(f"**{s['citation']}**")
                st.markdown(f"> {s['excerpt']}")
    if res.get("seconds") is not None and status in ("answered", "refused"):
        extra = " · cached" if res.get("cached") else ""
        st.caption(f"{res['seconds']}s · model: {res.get('model') or '-'}{extra} · Information from the GDPR text, not legal advice.")


def queue(q):
    st.session_state["queued"] = q


st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_q", None)

with st.sidebar:
    st.header("GDPR Assistant")
    st.write("Answers come only from the official GDPR text. Every claim is checked against the exact wording and shown with article and page.")
    h = api_health()
    if h:
        st.success(f"API online · {h['chunks']} chunks · {h['device']}")
    else:
        st.error("API not reachable")
    st.subheader("Try these")
    for i, q in enumerate(EXAMPLES):
        st.button(q, key=f"ex{i}", on_click=queue, args=(q,), use_container_width=True)
    if st.button("New topic", use_container_width=True):
        st.session_state.last_q = None
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_q = None
        st.rerun()

st.title("⚖️ GDPR Assistant")
st.caption("Not legal advice.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render(m["res"])

typed = st.chat_input("Ask about the GDPR...")
question = st.session_state.pop("queued", None) or typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the GDPR and checking every quote..."):
            try:
                res = call_api(question, st.session_state.last_q)
            except requests.exceptions.ConnectionError:
                res = warn("The API is not reachable. Is it running?")
            except requests.exceptions.Timeout:
                res = warn("The answer took too long. Please try again.")
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code
                res = warn({401: "The API key is wrong or missing.", 503: "The API is still loading. Try again in a minute.",
                            422: "Please type a question of at least 3 characters."}.get(code, f"API error {code}."))
        render(res)
    st.session_state.messages.append({"role": "assistant", "res": res})
    if res.get("status") in ("answered", "refused"):
        st.session_state.last_q = res.get("searched_as") or question