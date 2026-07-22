# ============================================================
# DAY 3 LAB — SKELETON: From Prototype to Enterprise
# (covers Day 3: multi-agent systems + Day 5: production agents)
# ============================================================
# Fill in every TODO. Each step tells you exactly WHERE in the
# docs to look. Don't copy from the solution file
# (../lab_prototype_to_enterprise.py) until you've tried each
# step — the point of this lab is learning what separates a
# DEMO from a PRODUCT.
#
# The system you're building — a multi-agent report generator
# (Day 3) that you then harden into an enterprise service (Day 5):
#
#   START → research → summarize → write → review
#                        ↑                   │
#                        └─ score < 8 ───────┤  (max 2 revisions)
#                                            └─ score >= 8 → END
#
#   ...then wrapped, layer by layer, in:
#
#   ┌─ Stage 5: FastAPI service (/health, /report) ──────────┐
#   │ ┌─ Stage 4: guardrails + cost budget ────────────────┐ │
#   │ │ ┌─ Stage 3: structured logs, run_id, latency ────┐ │ │
#   │ │ │ ┌─ Stage 2: config from env, secrets in .env ─┐│ │ │
#   │ │ │ │ ┌─ Stage 1: retries, backoff, timeouts ────┐││ │ │
#   │ │ │ │ │        Stage 0: the agent graph          │││ │ │
#   │ │ │ │ └──────────────────────────────────────────┘││ │ │
#   │ │ │ └─────────────────────────────────────────────┘│ │ │
#   │ │ └────────────────────────────────────────────────┘ │ │
#   │ └──────────────────────────────────────────────────────┘ │
#   └──────────────────────────────────────────────────────────┘
#
# Recommended reading BEFORE you start (~30 min):
#   1. Multi-agent concepts (supervisor pattern — today's graph):
#      https://docs.langchain.com/oss/python/langgraph/multi-agent
#   2. Graph API (you know this from Day 2 — skim as refresher):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api
#   3. Anthropic, "Building effective agents" (when NOT to
#      multi-agent): https://www.anthropic.com/research/building-effective-agents
#
# Model setup: same as Day 2 — OpenAI key, or OpenRouter free
# models (see the big OpenRouter block in
# ../../Day_2/Day Two Lab/Updated_2026/skeleton_research_agent.py,
# Step 2). No key at all? Set MOCK=1 and a fake model is used.
#
# Setup:
#   pip install langchain-openai langgraph python-dotenv fastapi uvicorn
# ============================================================

import json
import logging
import operator
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# TODO STEP 0 — import StateGraph, START, END from langgraph.graph
# (same imports as Day 2). #done above

load_dotenv()

STAGE = int(os.getenv("LAB_STAGE", "4"))   # 0..5 — your maturity level
MOCK = os.getenv("MOCK", "0") == "1"


"""
due to persistent errors, the settings class has been pushed up 
"""
@dataclass
class Settings:
    model_name: str = 'google/gemma-4-26b-a4b-it:free'
    temperature: float = 0.3
    request_timeout_s: int = 60
    max_retries: int = 3
    quality_threshold: int = 8
    max_revisions: int = 2
    cost_budget_usd: float = 0.25
    max_topic_len: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            request_timeout_s=int(os.getenv("REQUEST_TIMEOUT_S", cls.request_timeout_s)),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            quality_threshold=int(os.getenv("QUALITY_THRESHOLD", cls.quality_threshold)),
            max_revisions=int(os.getenv("MAX_REVISIONS", cls.max_revisions)),
            cost_budget_usd=float(os.getenv("COST_BUDGET_USD", cls.cost_budget_usd)),
            max_topic_len=int(os.getenv("MAX_TOPIC_LEN", cls.max_topic_len)),
        )


settings = Settings.from_env() if STAGE >= 2 else Settings()

PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60

# ============================================================
# STEP 1 — THE STATE (the contract between your agents)
# ============================================================
# Day 3 slides: agents coordinate through a COMMUNICATION
# MECHANISM — here it's shared graph state.
#
# Define a TypedDict with:
#   run_id (str), topic (str), research_notes (str), summary (str),
#   draft (str), review_feedback (str), score (int),
#   revision_count (int), tokens_in (int), tokens_out (int),
#   cost_usd (float), error (str),
#   execution_logs — with the operator.add REDUCER (Day 2!)
#
# ASK YOURSELF: why must revision_count live in STATE and not in
# a Python variable next to the graph? (Hint: checkpointing,
# multiple runs, serving this graph from an API later.)

class ReportState(TypedDict, total=False):
    run_id: str
    topic: str
    research_notes: str
    summary: str
    draft: str
    review_feedback: str
    score: int
    revision_count: int
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    error: str
    pass


# ============================================================
# STEP 2 — MODEL (with an offline mock)
# ============================================================
# Create the model exactly as in Day 2 (ChatOpenAI, or OpenRouter
# with base_url + :free model). ONE addition for Stage 1+:
#   pass  timeout=60, max_retries=0  to ChatOpenAI.
# max_retries=0?! Yes — YOU will own retries in Step 5, and two
# competing retry layers multiply (3 SDK x 3 yours = 9 calls).
#
# The FakeChatModel below lets everyone run the lab offline.
# Read it — note how it fails the first review on purpose so the
# revision loop always fires.

class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 200, "output_tokens": 300}


class FakeChatModel:
    def __init__(self):
        self.review_calls = 0

    def invoke(self, prompt: str):
        time.sleep(0.2)
        p = prompt.lower()
        if "reviewer" in p:
            self.review_calls += 1
            score = 6 if self.review_calls == 1 else 9
            return FakeResponse(f"SCORE: {score}\nFEEDBACK: Add a concrete example.")
        if "research" in p:
            return FakeResponse("- fact one\n- fact two\n- fact three")
        if "summar" in p:
            return FakeResponse("A concise summary of the research notes.")
        return FakeResponse("INTRODUCTION\n...\n\nBODY\n" + "Substantive findings. " * 20
                            + "\n\nCONCLUSION\n...")


# TODO: model = FakeChatModel() if MOCK else ChatOpenAI(...)

def get_model():
    if MOCK:
        return FakeChatModel()
    else:
        return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        request_timeout=settings.request_timeout_s  
            )

model = get_model()

# ============================================================
# STEP 3 — ROLE-SPECIALIZED AGENTS (Day 3, slides 28-39)
# ============================================================
# Four nodes, four ROLES: Researcher, Summarizer, Writer, Reviewer.
# Each is a plain function: state in → partial dict out (Day 2 rule).
#
# For now call the model DIRECTLY (model.invoke(prompt).content).
# In Step 5 you will refactor every call to go through ONE
# chokepoint — notice how painful it would be if you had 20 nodes.
#
# WRITER: if state has review_feedback, append it to the prompt
# ("A reviewer said: ... address this feedback"). That single line
# is what turns the loop from "retry" into genuine COLLABORATION.
#
# REVIEWER: force the format "SCORE: <n>\nFEEDBACK: <line>" and
# parse with re.search(r"SCORE:\s*(\d+)", ...). (Day 2 taught the
# better way — with_structured_output. BONUS: use it here too.)

def research_node(state: ReportState):
    usage: dict = {}
    notes = call_llm(
        f"You are a research agent. Produce detailed, factual research notes "
        f"as bullet points about: {state['topic']}",
        node="research", state=state, usage=usage,
    )
    return {"research_notes": notes, **usage}



def summarize_node(state: ReportState):
    usage: dict = {}
    summary = call_llm(
        f"You are a summarization agent. Summarize the following research notes "
        f"as a concise paragraph: {state['research_notes']}",
        node="summarize", state=state, usage=usage,
    )
    return {"summary": summary, **usage}


def write_node(state: ReportState):
    usage: dict = {}
    feedback = state.get("review_feedback", "")
    hint = f"\n\nA reviewer said: {feedback}. Address this feedback." if feedback else ""
    draft = call_llm(
        f"You are a writing agent. Create a structured report (introduction, body, "
        f"conclusion) based on the summary: {state['summary']}{hint}",
        node="write", state=state, usage=usage,
    )
    return {"draft": draft, **usage}


def review_node(state: ReportState):
    usage: dict = {}
    verdict = call_llm(
        f"You are a strict quality reviewer. Score this report from 1 to 10 "
        f"and give one line of feedback. Reply EXACTLY in this format:\n"
        f"SCORE: <number>\nFEEDBACK: <one line>\n\nReport:\n{state['draft']}",
        node="review", state=state, usage=usage,
    )
    score_match = re.search(r"SCORE:\s*(\d+)", verdict)
    score = int(score_match.group(1)) if score_match else 0
    fb_match = re.search(r"FEEDBACK:\s*(.+)", verdict)
    feedback = fb_match.group(1).strip() if fb_match else verdict
    revision = state.get("revision_count", 0) + 1

    if STAGE >= 3:
        log_event("review_verdict", run_id=state.get("run_id", "-"),
                  score=score, revision=revision)
    return {
        "score": score,
        "review_feedback": feedback,
        "revision_count": revision,
        **usage,
    }

# ============================================================
# STEP 4 — THE SUPERVISOR DECISION (Day 3: coordination strategy)
# ============================================================
# Router after review:
#   "approve"  score >= QUALITY_THRESHOLD (8)
#   "give_up"  revision_count > MAX_REVISIONS (2)   <- Day 2 lesson:
#                                                      loops MUST terminate
#   "revise"   otherwise → back to write
#
# Then wire the graph:
#   START → research → summarize → write → review
#   add_conditional_edges("review", review_gate,
#       {"approve": END, "give_up": END, "revise": "write"})
#
# WHERE TO LOOK: same conditional-branching docs as Day 2.
# ASK YOURSELF: why does "revise" go to write, not research?
# When WOULD you route back to research instead?

def review_gate(state: ReportState) -> str:
    if state["score"] >= settings.quality_threshold:
        return "approve"
    if state["revision_count"] > settings.max_revisions:
        return "give_up"
    return "revise"


def build_graph():
    g = StateGraph(ReportState)
    g.add_node("research", research_node)
    g.add_node("summarize", summarize_node)
    g.add_node("write", write_node)
    g.add_node("review", review_node)

    g.add_edge(START, "research")
    g.add_edge("research", "summarize")
    g.add_edge("summarize", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges(
        "review", review_gate,
        {"approve": END, "give_up": END, "revise": "write"},
    )
    return g.compile()


graph = build_graph()
# TODO: build + compile the graph  →  graph = workflow.compile()


# ============================================================
# ============================================================
#   YOU ARE NOW HERE: a working PROTOTYPE (Stage 0).
#   Everything below is Day 5 — crossing the PoC chasm.
#   Each stage guards its code with  `if STAGE >= n:`  so one
#   file can demonstrate every maturity level.
# ============================================================
# ============================================================


# ============================================================
# STEP 5 — STAGE 1: ROBUSTNESS (Day 5: "Error Handling")
# ============================================================
# Refactor: every node now calls  call_llm(prompt, node, state)
# instead of model.invoke. Implement it with:
#   - up to MAX_RETRIES attempts
#   - exponential backoff WITH JITTER between attempts:
#       delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
#   - on final failure: raise RuntimeError with node name + error
#   - in generate_report (Step 9): catch it and return a partial
#     result with state["error"] set — degrade, don't crash.
#
# WHERE TO LOOK: https://docs.aws.amazon.com/general/latest/gr/api-retries.html
#   (the canonical backoff+jitter explanation — 5 min read)
# TEST IT: temporarily add
#   if random.random() < 0.3: raise TimeoutError("boom")
# before the invoke and watch retries fire.
# ASK YOURSELF: why jitter? What happens when 100 replicas all
# retry at exactly t=1s, 2s, 4s?

def call_llm(prompt: str, node: str, state: ReportState, usage: dict) -> str:
    # فحص الميزانية قبل الصرف
    if STAGE >= 4:
        spent = state.get("cost_usd", 0.0) + usage.get("cost_usd", 0.0)
        if spent >= settings.cost_budget_usd:
            raise BudgetExceeded(
                f"Cost budget ${settings.cost_budget_usd} exhausted before node '{node}'"
            )

    attempts = settings.max_retries if STAGE >= 1 else 1
    last_err = None

    for attempt in range(1, attempts + 1):
        try:
            response = model.invoke(prompt)

            meta = getattr(response, "usage_metadata", None) or {}
            t_in = meta.get("input_tokens", len(prompt) // 4)
            t_out = meta.get("output_tokens", len(response.content) // 4)
            usage["tokens_in"] = usage.get("tokens_in", 0) + t_in
            usage["tokens_out"] = usage.get("tokens_out", 0) + t_out
            usage["cost_usd"] = round(
                usage.get("cost_usd", 0.0)
                + t_in * PRICE_IN_PER_M / 1e6
                + t_out * PRICE_OUT_PER_M / 1e6,
                6,
            )

            if STAGE >= 3:
                log_event("llm_call", run_id=state.get("run_id", "-"),
                          node=node, attempt=attempt, tokens_in=t_in, tokens_out=t_out)
            return response.content

        except Exception as exc:
            last_err = exc
            if attempt == attempts:
                break
            delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
            if STAGE >= 3:
                log_event("llm_retry", run_id=state.get("run_id", "-"),
                          node=node, attempt=attempt, error=str(exc)[:120])
            time.sleep(delay)

    raise RuntimeError(f"Node '{node}' failed after {attempts} attempt(s): {last_err}")

    pass


# ============================================================
# STEP 6 — STAGE 2: CONFIG & SECRETS (Day 5: "Security & Governance")
# ============================================================
# Kill every hardcoded number. Build a Settings dataclass:
#   model_name, temperature, request_timeout_s, max_retries,
#   quality_threshold, max_revisions, cost_budget_usd, max_topic_len
# with a  from_env()  classmethod reading os.getenv with defaults.
#
#   settings = Settings.from_env() if STAGE >= 2 else Settings()
#
# WHERE TO LOOK: https://12factor.net/config  (10 min, classic)
# PROVE IT WORKS:  QUALITY_THRESHOLD=10 LAB_STAGE=2 python ...
# → the reviewer can never approve → give_up path fires. No code
# edits. That's the point.




# ============================================================
# STEP 7 — STAGE 3: OBSERVABILITY (Day 5: "Observability & Maintenance")
# ============================================================
# print() doesn't survive contact with production. Emit ONE JSON
# object per event so a log platform can index and query them:
#   {"ts": ..., "level": ..., "event": "llm_call", "run_id": ...,
#    "node": "write", "attempt": 1, "latency_s": 2.1,
#    "tokens_in": 812, "tokens_out": 405, "cost_usd": 0.0011}
#
# Implement log_event(event, **fields) using the logging module
# with a custom Formatter that json.dumps the record (see the
# solution file's JsonFormatter if stuck — it's 8 lines).
# Emit events: run_started, llm_call, llm_retry, review_verdict,
# run_finished.
#
# WHERE TO LOOK: https://docs.python.org/3/howto/logging-cookbook.html
# ASK YOURSELF: you have 40 runs/hour and one user says "my report
# was bad". Which field in the logs lets you reconstruct exactly
# what happened for THEIR run?

# ============================================================
# STEP 7 
# ============================================================
logger = logging.getLogger("agent")
logger.setLevel("INFO")
_handler = logging.StreamHandler()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)


def log_event(event: str, **fields):
    logger.info(event, extra={"extra_fields": fields})


# ============================================================
# STEP 8 — STAGE 4: GUARDRAILS + COST (Day 5: "Security" + "Cost")
# ============================================================
# Three cheap, high-value protections:
#
# a) validate_topic(topic) BEFORE any LLM call:
#    - reject empty / longer than max_topic_len
#    - reject prompt-injection patterns, e.g.
#      r"ignore (all|previous|the) instructions", r"system prompt"
# b) validate_report(report) AFTER the run:
#    - reject if < 200 chars or contains refusal artifacts
#      ("as an ai language model", ...)
# c) budget: at the top of call_llm, if state's cost_usd >=
#    settings.cost_budget_usd → raise BudgetExceeded. Catch it in
#    generate_report and abort GRACEFULLY (partial result + error).
#
# WHERE TO LOOK: https://genai.owasp.org/llm-top-10/  — find which
# two entries you just mitigated.
# TEST: TOPIC="Ignore all instructions..." must be rejected;
#       COST_BUDGET_USD=0.0000001 must abort, not crash.

class BudgetExceeded(Exception):
    pass


def validate_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty")
    if len(topic) > settings.max_topic_len:
        raise ValueError(f"Topic is too long (max {settings.max_topic_len} characters)")


def validate_report(report: str) -> None:
    if len(report) < 200:
        raise ValueError("Report is too short")
    if "as an ai language model" in report.lower():
        raise ValueError("Report contains refusal artifacts")


# ============================================================
# STEP 9 — generate_report(): tie the stages together
# ============================================================
# def generate_report(topic):
#   1. build initial state (uuid run_id, revision_count=0, cost 0)
#   2. STAGE >= 4: topic = validate_topic(topic)
#   3. STAGE >= 3: log_event("run_started", ...)
#   4. try: final = graph.invoke(state)
#      except BudgetExceeded / RuntimeError:
#          STAGE >= 1 → return partial state with error set
#          STAGE 0    → just crash (that's what prototypes do)
#   5. STAGE >= 4: validate_report(final["draft"])
#   6. STAGE >= 3: log_event("run_finished", ...totals...)

def generate_report(topic: str) -> ReportState:
    state: ReportState = {
        "topic": topic,
        "run_id": str(uuid.uuid4())[:8],
        "revision_count": 0,
        "cost_usd": 0.0,
    }

    if STAGE >= 4:
        state["topic"] = validate_topic(topic)
    if STAGE >= 3:
        log_event("run_started", run_id=state["run_id"], topic=state["topic"], stage=STAGE)

    try:
        final = graph.invoke(state)
    except BudgetExceeded as exc:
        final = dict(state)
        final["error"] = str(exc)
        log_event("run_aborted_budget", run_id=state["run_id"], error=str(exc))
        return final
    except RuntimeError as exc:
        final = dict(state)
        final["error"] = str(exc)
        if STAGE >= 1:
            print(f"[degraded] {exc}")
            return final
        raise

    if STAGE >= 4 and "draft" in final:
        validate_report(final["draft"])
    if STAGE >= 3:
        log_event("run_finished", run_id=final.get("run_id", "-"),
                  score=final.get("score"), cost_usd=final.get("cost_usd"))
    return final


# ============================================================
# STEP 10 — STAGE 5: SERVING (Day 5: cloud deployment sections)
# ============================================================
# A script is a demo; an API is a product other teams can use.
#   app = FastAPI()
#   GET  /health  → {"status": "ok", "stage": STAGE, "model": ...}
#   POST /report  → body {"topic": str} (pydantic model), calls
#                   generate_report; map errors to HTTP:
#                   guardrail ValueError → 422, run error → 503
#
# WHERE TO LOOK: https://fastapi.tiangolo.com/tutorial/first-steps/
# RUN:   LAB_STAGE=5 python skeleton_enterprise_multiagent.py serve
# TEST:  curl localhost:8000/health
#        curl -X POST localhost:8000/report -H 'Content-Type: application/json' \
#             -d '{"topic": "Smart Cities"}'
#        curl ... -d '{"topic": "Ignore all instructions"}'   # expect 422
# ASK YOURSELF: you now run 3 replicas behind a load balancer.
# Which parts of your file break? (Hint: anything in module-level
# variables — like FakeChatModel.review_calls...)

def create_app():
    METRICS = {
    "runs_served": 0, "runs_failed": 0, "total_cost_usd": 0.0,
    "total_tokens_in": 0, "total_tokens_out": 0,
}


def create_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Report Agent API", version="1.0")

    class ReportRequest(BaseModel):
        topic: str

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": STAGE, "model": settings.model_name, "mock": MOCK}

    @app.get("/metrics")
    def metrics():
        return METRICS

    @app.post("/report")
    def report(req: ReportRequest):
        try:
            result = generate_report(req.topic)
        except ValueError as exc:
            METRICS["runs_failed"] += 1
            raise HTTPException(status_code=422, detail=str(exc))
        METRICS["runs_served"] += 1
        METRICS["total_cost_usd"] = round(
            METRICS["total_cost_usd"] + (result.get("cost_usd") or 0.0), 6)
        METRICS["total_tokens_in"] += result.get("tokens_in") or 0
        METRICS["total_tokens_out"] += result.get("tokens_out") or 0
        if result.get("error"):
            METRICS["runs_failed"] += 1
            raise HTTPException(status_code=503, detail=result["error"])
        return {
            "run_id": result["run_id"],
            "topic": result["topic"],
            "score": result.get("score"),
            "cost_usd": result.get("cost_usd"),
            "report": result.get("draft"),
        }

    return app


if __name__ == "__main__":
    print(f"=== STAGE {STAGE} {'(MOCK)' if MOCK else ''} ===")
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    else:
        topic = os.getenv("TOPIC", "do whatever you want")
        result = generate_report(topic)
        print("\n--- REPORT ---")
        print(result.get("draft") or f"NO REPORT — {result.get('error')}")
        print(f"\nscore={result.get('score')} | cost=${result.get('cost_usd')} "
              f"| revisions={result.get('revision_count')}")

# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# Day 3 (the agent):
# [ ] Four role agents communicate ONLY through graph state
# [ ] The writer actually USES the reviewer's feedback on revision
# [ ] My loop has both a quality exit AND a revision cap (Day 2!)
# [ ] I can explain when I'd route "revise" → research instead of write
# Day 5 (the chasm):
# [ ] ALL model calls go through call_llm — zero direct invokes left
# [ ] I know why SDK max_retries=0 when I own retries (and why jitter)
# [ ] QUALITY_THRESHOLD=10 changes behavior with zero code edits
# [ ] My logs are one JSON object per line, every one has run_id
# [ ] Injection topic → rejected BEFORE any money is spent
# [ ] Budget exhaustion aborts gracefully mid-run (partial + error)
# [ ] /report returns 422 for guardrail hits, 503 for run failures
# [ ] I can name 3 things STILL missing for real production
#     (auth? rate limiting? queue for long runs? containers? CI/CD?)
#
# Stuck? Debugging order that works:
#   1. MOCK=1 LAB_STAGE=0 — get the bare graph green first
#   2. raise a fake TimeoutError inside call_llm — watch retries
#   3. pipe Stage 3 output through `python -m json.tool` per line
#   4. only THEN open ../lab_prototype_to_enterprise.py
# ============================================================
