# ============================================================
# DAY 2 LAB — COMPLETED: Enterprise Research Agent (LangGraph)
# ============================================================
# Flow:
#   START → collect → store_memory → analyze → evaluate
#              ↑                                  │
#              └── quality < 7 (max 3 tries) ─────┤
#                                                 └ quality >= 7
#                                                       ↓
#                                          report → audit → END
# ============================================================


# ============================================================
# STEP 0 — IMPORTS
# ============================================================
import os
import operator
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_community.embeddings import HuggingFaceEmbeddings
from html import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# Loop-guard constants (used by the conditional edge in Step 5)
QUALITY_THRESHOLD = 7
MAX_RESEARCH_ITERATIONS = 3


# ============================================================
# STEP 1 — THE STATE  (the "digital clipboard")
# ============================================================
# execution_logs uses a REDUCER (operator.add): every node returns
# only its NEW log lines and LangGraph appends them. A plain key
# would be OVERWRITTEN by the last node that writes it.

class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]



# ============================================================
# STEP 2 — MODEL, SEARCH/FETCH TOOLS, EMBEDDINGS
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-it:free"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

search_tool = TavilySearch(max_results=5)      
extract_tool = TavilyExtract()                 

research_tools = [search_tool, extract_tool]
llm_with_tools = llm.bind_tools(research_tools)
TOOLS_BY_NAME = {tool.name: tool for tool in research_tools}

COLLECT_SYSTEM_PROMPT = (
    "You are a research assistant gathering sources on a topic. You have "
    "two tools:\n"
    f"- {search_tool.name}: run a web search for a query and get back a "
    "list of matching pages with short content snippets. Use this to "
    "DISCOVER sources on the topic.\n"
    f"- {extract_tool.name}: fetch the full content of one or more "
    "specific URLs you already know about — from a prior search result, "
    "or one the user gave you directly. Use this when a snippet isn't "
    "enough and you need the full page text.\n\n"
    "Call one or more tools to gather strong sources for the given "
    "research query. Prefer searching first to discover sources, then "
    "extracting the most promising URLs if you need deeper content. "
    "Always call at least one tool — do not answer from memory."
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma = persistent vector memory across runs.
try:
    vector_store = Chroma(
        collection_name="enterprise_research_memory",
        embedding_function=embedding_model,
        persist_directory="./enterprise_memory_db",
    )
    print("[setup] Using Chroma (persistent vector memory).")
except ImportError:
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    print("[setup] chromadb not installed — using in-memory vector store.")


# ============================================================
# STEP 3 — STRUCTURED OUTPUT for the quality score
# ============================================================

class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10, description="Overall research quality, 1-10")
    reasoning: str = Field(description="One-sentence justification")


evaluator = llm.with_structured_output(QualityScore)

# Optional sanity check (uncomment to test the evaluator alone):
# test = evaluator.invoke(
#     "Rate the quality of this claim from 1-10: "
#     "'The Earth orbits the Sun, supported by centuries of evidence.'"
# )
# print(type(test), test.score, test.reasoning)


# Small helper: timestamped log line, printed AND returned as a
# one-element list ready for the execution_logs reducer.
def log(message: str) -> List[str]:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line)
    return [line]


def _normalize_tool_output(output: Dict) -> List[Dict]:
    """Flatten a Tavily tool's raw output into the {"url", "content"}
    shape the rest of the graph (analyze_node, etc.) already expects.
    search_tool results carry "content"; extract_tool results carry
    "raw_content" instead — this reconciles the two."""
    items = output.get("results", []) if isinstance(output, dict) else []
    return [
        {
            "url": item.get("url", "unknown"),
            "content": item.get("content") or item.get("raw_content", ""),
        }
        for item in items
    ]


# ============================================================
# STEP 4 — NODES
# ============================================================
# A node takes state and returns a PARTIAL update (only the keys
# it changed). LangGraph merges it in — never mutate state in place.

def collect_node(state: AgentState):
    """Ask the LLM to gather sources — it decides whether to run a
    keyword search, fetch a specific URL's full content, or both,
    via the tools bound in Step 2. On retries, CHANGE the query so
    the model has a real chance to turn up different sources."""
    iteration = state["iteration_count"] + 1

    # LOOP GUARD, part 1: each retry uses a meaningfully different
    # query — different keywords → different sources → a real chance
    # the quality score improves on the next pass.
    refinements = [
        state["topic"],
        f"{state['topic']} latest developments case studies",
        f"{state['topic']} industry analysis best practices",
    ]
    # Clamp the index so extra iterations reuse the last refinement
    # instead of raising an IndexError.
    query = refinements[min(iteration - 1, len(refinements) - 1)]

    ai_message = llm_with_tools.invoke([
        SystemMessage(COLLECT_SYSTEM_PROMPT),
        HumanMessage(f"Research query: {query}"),
    ])

    results: List[Dict] = []
    for call in ai_message.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            continue
        results.extend(_normalize_tool_output(tool.invoke(call["args"])))

    called = ", ".join(c["name"] for c in ai_message.tool_calls) or "none"
    logs = log(
        f"Iteration {iteration}: LLM called [{called}] and collected "
        f"{len(results)} sources for query: '{query}'"
    )

    if not results:
        # Local models can occasionally skip tool calls or return
        # empty results — fall back to a direct search so the
        # pipeline never stalls with zero sources.
        logs += log("No sources from tool calls — falling back to a direct search.")
        results = _normalize_tool_output(search_tool.invoke({"query": query}))

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": logs,
    }


def store_memory_node(state: AgentState):
    """Save source contents into the vector store."""
    contents = [source.get("content", "")
                for source in state["collected_data"]
                if source.get("content")]
    if contents:
        vector_store.add_texts(contents)

    return {
        "execution_logs": log(
            f"Stored {len(contents)} documents in vector memory."),
    }


def analyze_node(state: AgentState):
    """LLM-analyze each source, enriched with related past research
    retrieved from the vector store — that's what makes this RAG."""
    analyzed = []

    for source in state["collected_data"]:
        content = source.get("content", "")

        # RAG: pull related past research from vector memory
        related = vector_store.similarity_search(content, k=2)
        related_text = "\n".join(doc.page_content for doc in related)

        prompt = (
            "Analyze this research source and extract the key findings.\n\n"
            f"SOURCE:\n{content}\n\n"
            f"RELATED PAST RESEARCH (for context):\n{related_text}\n\n"
            "Give a concise analysis of the main points."
        )
        response = llm.invoke(prompt)

        analyzed.append({
            "source": source.get("url", "unknown"),
            "analysis": response.content,
        })

    return {
        "analyzed_data": analyzed,
        "execution_logs": log(
            f"Analyzed {len(analyzed)} sources with RAG context."),
    }


def evaluate_node(state: AgentState):
    """Score the research with the STRUCTURED evaluator (Step 3)."""
    summary = "\n\n".join(item["analysis"] for item in state["analyzed_data"])

    result = evaluator.invoke(
        "Rate the overall quality of this research analysis "
        f"on a scale of 1-10.\n\n{summary}"
    )

    return {
        "quality_score": result.score,
        "execution_logs": log(
            f"Quality score = {result.score} ({result.reasoning})"),
    }


def report_node(state: AgentState):
    """Generate the enterprise report from analyzed_data."""
    summary = "\n\n".join(
        f"Source: {item['source']}\n{item['analysis']}"
        for item in state["analyzed_data"]
    )

    prompt = (
        f"Write a professional research report on the topic: {state['topic']}.\n\n"
        f"Base it on the following analyzed sources:\n\n{summary}\n\n"
        "Structure it with a summary, key findings, and a conclusion."
    )
    response = llm.invoke(prompt)

    return {
        "final_report": response.content,
        "execution_logs": log("Final report generated."),
    }


def audit_node(state: AgentState):
    """Log completion stats."""
    return {
        "execution_logs": log(
            f"AUDIT COMPLETE — topic: {state['topic']}, "
            f"iterations: {state['iteration_count']}, "
            f"final score: {state['quality_score']}, "
            f"sources analyzed: {len(state['analyzed_data'])}, "
            f"report length: {len(state['final_report'])} chars"
        ),
    }


# ============================================================
# STEP 5 — THE CONDITIONAL EDGE (the heart of this lab)
# ============================================================
# Returns the NAME of the next node as a string. Loops terminate
# because of TWO guards:
#   a) every retry changes the query (collect_node, Step 4),
#   b) a hard cap on iteration_count (here).
# Without both: same search → same score → infinite loop →
# GraphRecursionError at recursion limit 25.

def quality_router(state: AgentState) -> str:
    score = state["quality_score"]
    iteration = state["iteration_count"]

    if score >= QUALITY_THRESHOLD:
        print(f"[router] quality {score} >= {QUALITY_THRESHOLD} -> report")
        return "report"

    # LOOP GUARD, part 2: hard cap on retries.
    if iteration >= MAX_RESEARCH_ITERATIONS:
        print(f"[router] max iterations ({iteration}) reached -> report anyway")
        return "report"

    print(f"[router] quality {score} < {QUALITY_THRESHOLD} -> recollecting")
    return "collect"


# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "collect")
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

# The dict maps router RETURN VALUES to NODE NAMES.
workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {
        "collect": "collect",
        "report": "report",
    },
)

workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)


# ============================================================
# STEP 7 — COMPILE with a checkpointer, VISUALIZE, RUN
# ============================================================
# The checkpointer saves state after every node → enables resume,
# time-travel debugging, and human-in-the-loop.

checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    # BONUS — human-in-the-loop: uncomment to pause before the
    # report so you can inspect state and resume manually:
    # interrupt_before=["report"],
)

# Visualize (paste the output into https://mermaid.live):
print("\n--- GRAPH STRUCTURE (Mermaid) ---")
print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    initial_state = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    # thread_id is REQUIRED when a checkpointer is attached.
    config = {"configurable": {"thread_id": "lab-day2-run-1"}}

    # Stream with stream_mode="values" so we watch the full state
    # evolve node by node; the last chunk is the final state.
    final_state = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final_state = chunk

    doc = SimpleDocTemplate("final_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for para in final_state["final_report"].split("\n"):
        if para.strip():
            story.append(Paragraph(escape(para.strip()), styles["Normal"]))
            story.append(Spacer(1, 6))
    doc.build(story)
    print("Saved: final_report.pdf")
    print("\n================================================")
    print("EXECUTION LOGS")
    print("================================================")
    for line in final_state["execution_logs"]:
        print(line)
