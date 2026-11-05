import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from gdpr_rag import GDPRBot  # noqa: E402

API_KEY = os.getenv("API_KEY", "")
state = {"bot": None}
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bot"] = GDPRBot()          # loads the data and the two AI models once, at start-up
    yield


app = FastAPI(title="GDPR Assistant API", version="1.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    previous_question: Optional[str] = Field(default=None, max_length=500)


class Claim(BaseModel):
    text: str
    citations: List[str]


class Source(BaseModel):
    citation: str
    excerpt: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class AskResponse(BaseModel):
    status: str                      # answered | refused | warning
    answer: str
    claims: List[Claim] = []
    sources: List[Source] = []
    removed: int = 0
    model: Optional[str] = None
    searched_as: str = ""
    cached: bool = False
    seconds: float = 0.0
    timings: Dict[str, float] = {}


def get_bot(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")
    if state["bot"] is None:
        raise HTTPException(503, "Still loading")
    return state["bot"]


@app.get("/")
def root():
    return {"name": "GDPR Assistant API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    bot = state["bot"]
    if bot is None:
        raise HTTPException(503, "Still loading")
    return bot.info()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    bot = get_bot(x_api_key)
    with lock:                        # one question at a time keeps CPU and memory predictable
        return bot.ask(req.question, req.previous_question)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    """Sends one JSON line per event: progress steps first, then the final result."""
    bot = get_bot(x_api_key)

    def events():
        try:
            with lock:
                for ev in bot.ask_events(req.question, req.previous_question):
                    if ev["type"] == "result":
                        ev = {"type": "result", "result": AskResponse(**ev["result"]).model_dump()}
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"[error] {e}", flush=True)
            err = AskResponse(status="warning", answer="Something went wrong on the server. Please try again.")
            yield json.dumps({"type": "result", "result": err.model_dump()}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})