# agent.py — Refined multi-agent pipeline (Research -> Writer -> Editor -> SEO)
# Uses Tavily for research (if available) and Gemini (via llm) for writing/editing.
# Provides robust fallbacks and sanitization so UI never shows raw Gemini error traces.

import os
import time
import random
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# Try import crewai & tavily tool (optional)
crew_available = False
try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import TavilySearchTool
    crew_available = True
except Exception:
    crew_available = False

# genai client (google-genai)
genai_client = None
try:
    from google import genai
    if GEMINI_KEY:
        # Some google-genai versions auto-pick up env var; others accept api_key
        try:
            genai_client = genai.Client(api_key=GEMINI_KEY)
        except Exception:
            # fallback to default client (which reads env)
            genai_client = genai.Client()
except Exception:
    genai_client = None

# load optional samples.json (demo)
SAMPLES = {}
try:
    SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "samples.json")
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            SAMPLES = json.load(f)
except Exception:
    SAMPLES = {}

# ---- LLM wrapper (retries + model fallback) ----
class GeminiLLM:
    def __init__(self, models=None):
        if models is None:
            self._models = ["gemini-2.5-flash", "gemini-2.0-pro", "gemini-2.0-flash"]
        else:
            self._models = models

    def _call_once(self, model_name: str, prompt: str):
        if genai_client is None:
            raise RuntimeError("genai client not initialized.")
        return genai_client.models.generate_content(model=model_name, contents=prompt)

    def run(self, prompt: str, max_retries: int = 2) -> str:
        if genai_client is None:
            return "GEMINI_RESOURCE_EXHAUSTED"
        for model in self._models:
            retry = 0
            while retry <= max_retries:
                try:
                    resp = self._call_once(model, prompt)
                    if hasattr(resp, "text") and resp.text:
                        return resp.text
                    try:
                        # SDK variants: resp.candidates -> content.parts...
                        return resp.candidates[0].content.parts[0].text
                    except Exception:
                        return str(resp)
                except Exception as e:
                    err = str(e).lower()
                    # transient or quota -> retry
                    if any(tok in err for tok in ("429", "resource_exhausted", "503", "unavailable", "rate limit", "overload")):
                        wait = (2 ** retry) + random.random()
                        print(f"[GeminiLLM] transient error for {model} (attempt {retry+1}): {e}; waiting {wait:.1f}s")
                        time.sleep(wait)
                        retry += 1
                        continue
                    # other errors -> return marker
                    print(f"[GeminiLLM] non-retryable error for {model}: {e}")
                    return f"[Gemini Error] {e}"
            print(f"[GeminiLLM] model {model} exhausted after retries.")
        return "GEMINI_RESOURCE_EXHAUSTED"

llm = GeminiLLM()

# ---- Tavily wrapper (safe) ----
tavily_tool = None
if crew_available:
    try:
        tavily_tool = TavilySearchTool(api_key=TAVILY_KEY) if TAVILY_KEY else None
    except Exception:
        tavily_tool = None

# ---- Helpers ----
def sanitize_text_output(text: Any, topic: str = None, prediction: dict = None) -> str:
    """Replace raw Gemini error traces with friendly fallback summary."""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""
    low = text.lower()
    error_signals = ["resource_exhausted", "resource exhausted", "429", "503", "unavailable",
                     "rate limit", "rate_limit", "overload", "overloaded", "[gemini error]", "not_found", "404"]
    if any(tok in low for tok in error_signals):
        if prediction is None:
            prediction = {"signal": "Unknown", "confidence": 0.0, "reason": "No reason available"}
        topic = topic or "the topic"
        return fallback_summary(topic, prediction)
    return text

def fallback_summary(topic: str, prediction: dict) -> str:
    sig = prediction.get("signal", "Unknown")
    conf = prediction.get("confidence", 0.0)
    reason = prediction.get("reason", "No reason available")
    return (f"Offline summary for {topic}:\n\n"
            f"- Signal: {sig} (confidence {conf:.2f})\n"
            f"- Reason: {reason}\n\n"
            "Note: Cloud LLM temporarily unavailable (quota/billing). This offline summary is shown so the demo remains operational.")

def sample_for_topic(topic: str) -> Dict[str, Any]:
    if not topic:
        return {}
    key = topic.strip().lower()
    if key in SAMPLES:
        return SAMPLES[key]
    # simple normalization
    key2 = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in key).split())
    if key2 in SAMPLES:
        return SAMPLES[key2]
    # substring
    for k in SAMPLES:
        if k in key or key in k:
            return SAMPLES[k]
    return {}

# ---- Research agent ----
def run_research(topic: str, logs: List[str]) -> Tuple[str, List[str]]:
    """
    Robust research runner that supports:
      - tavily_client (standalone SDK)
      - tavily_tool wrappers (crewai_tools.TavilySearchTool or similar)
      - fallback to samples.json when live search is unavailable
    Returns (research_text, logs)
    """
    research_items: List[str] = []

    # DEMO sample override
    if DEMO_MODE:
        sample = sample_for_topic(topic)
        if sample and isinstance(sample.get("pipeline"), dict):
            logs.append("DEMO_MODE: using sample research.")
            sample_research = sample.get("research") or sample["pipeline"].get("ResearchAgent", "")
            return (sample_research or ""), logs

    resp = None

    # 1) Try tavily_client (standalone SDK) if available
    if 'tavily_client' in globals() and tavily_client:  # type: ignore
        try:
            logs.append("Using tavily_client (standalone SDK) to search...")
            try:
                resp = tavily_client.search(query=topic, include_answer=True, max_results=6)  # type: ignore
            except TypeError:
                resp = tavily_client.search(topic)  # type: ignore
            logs.append("tavily_client returned a response.")
        except Exception as e:
            logs.append(f"tavily_client call failed: {e}")
            resp = None

    # 2) If no resp yet, try tavily_tool wrapper
    if not resp and 'tavily_tool' in globals() and tavily_tool:  # type: ignore
        try:
            logs.append("Attempting to use tavily_tool wrapper...")
            if hasattr(tavily_tool, "search"):
                resp = tavily_tool.search(topic)  # type: ignore
                logs.append("Called tavily_tool.search(...)")
            elif hasattr(tavily_tool, "run"):
                resp = tavily_tool.run(topic)  # type: ignore
                logs.append("Called tavily_tool.run(...)")
            elif hasattr(tavily_tool, "query"):
                resp = tavily_tool.query(topic)  # type: ignore
                logs.append("Called tavily_tool.query(...)")
            elif hasattr(tavily_tool, "search_query"):
                resp = tavily_tool.search_query(topic)  # type: ignore
                logs.append("Called tavily_tool.search_query(...)")
            elif callable(tavily_tool):  # type: ignore
                resp = tavily_tool(topic)  # type: ignore
                logs.append("Called tavily_tool(...) (callable)")
            else:
                logs.append("tavily_tool present but no known call method found.")
                resp = None
        except Exception as e:
            logs.append(f"tavily_tool call failed: {e}")
            resp = None

    # 3) Try inner client attribute if available
    if not resp and 'tavily_tool' in globals() and tavily_tool:  # type: ignore
        try:
            client_attr = getattr(tavily_tool, "client", None)  # type: ignore
            if client_attr:
                logs.append("Found tavily_tool.client attribute; attempting client.search(...)")
                try:
                    resp = client_attr.search(query=topic, include_answer=True, max_results=6)  # type: ignore
                    logs.append("Called tavily_tool.client.search(...)")
                except Exception:
                    try:
                        resp = client_attr.search(topic)  # type: ignore
                        logs.append("Called tavily_tool.client.search(topic)")
                    except Exception as e:
                        logs.append(f"tavily_tool.client.search failed: {e}")
            else:
                logs.append("No client attribute on tavily_tool to try.")
        except Exception as e:
            logs.append(f"Error while probing tavily_tool.client: {e}")

    # Parse resp (could be dict/list/object or JSON string)
    if resp:
        logs.append("Parsing tavily response into readable text...")
        items: List[str] = []
        parsed = None

        if isinstance(resp, str):
            try:
                parsed = json.loads(resp)
                logs.append("Decoded response string as JSON.")
            except Exception:
                parsed = resp
                logs.append("Response is a string but not JSON; will stringify.")
        else:
            parsed = resp

        try:
            # dict with 'results' / 'data' / 'items'
            if isinstance(parsed, dict):
                seq = None
                for key in ("results", "data", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        seq = parsed[key]
                        break
                if seq is None:
                    # maybe a single-result dict with title/url/content
                    if any(k in parsed for k in ("url", "title", "content", "raw_content", "answer", "summary")):
                        seq = [parsed]

                if seq:
                    for r in seq[:6]:
                        if isinstance(r, dict):
                            title = r.get("title") or r.get("headline") or ""
                            url = r.get("url") or r.get("link") or ""
                            snippet = r.get("content") or r.get("raw_content") or r.get("snippet") or r.get("summary") or r.get("text") or ""
                            score = r.get("score")
                            block = []
                            if title:
                                block.append(title.strip())
                            if url:
                                block.append(url.strip())
                            if snippet:
                                s = str(snippet).strip()
                                block.append(s if len(s) <= 300 else s[:300].rsplit(" ", 1)[0] + "...")
                            if score is not None:
                                block.append(f"[score: {score}]")
                            items.append("\n".join(block).strip())
                        else:
                            items.append(str(r))
                else:
                    # fallback to common single fields
                    title = parsed.get("title", "") if isinstance(parsed.get("title", ""), str) else ""
                    url = parsed.get("url", "") if isinstance(parsed.get("url", ""), str) else ""
                    snippet = parsed.get("content") or parsed.get("raw_content") or parsed.get("snippet") or parsed.get("summary") or ""
                    if title or url or snippet:
                        items.append("\n".join(p for p in (title, url, snippet) if p))
                    else:
                        items.append(json.dumps(parsed, indent=2)[:4000])
            elif isinstance(parsed, list):
                for r in parsed[:6]:
                    if isinstance(r, dict):
                        title = r.get("title") or r.get("headline") or ""
                        url = r.get("url") or r.get("link") or ""
                        snippet = r.get("content") or r.get("raw_content") or r.get("snippet") or r.get("summary") or ""
                        snippet_short = snippet if not snippet else (snippet if len(snippet) <= 300 else snippet[:300].rsplit(" ", 1)[0] + "...")
                        items.append("\n".join(p for p in (title, url, snippet_short) if p).strip())
                    else:
                        items.append(str(r))
            else:
                # object-like with results attr
                if hasattr(parsed, "results"):
                    seq = getattr(parsed, "results")
                    for r in list(seq)[:6]:
                        if isinstance(r, dict):
                            title = r.get("title","")
                            url = r.get("url","")
                            snippet = r.get("content") or r.get("raw_content") or r.get("snippet","")
                            items.append("\n".join(p for p in (title, url, snippet[:300]) if p).strip())
                        else:
                            title = getattr(r, "title", "") or getattr(r, "headline", "")
                            url = getattr(r, "url", "")
                            snippet = getattr(r, "content", "") or getattr(r, "raw_content", "") or getattr(r, "snippet", "")
                            items.append("\n".join(p for p in (title, url, snippet) if p).strip())
                else:
                    try:
                        items.append(json.dumps(parsed, default=str)[:4000])
                    except Exception:
                        items.append(str(parsed))
        except Exception as e:
            logs.append(f"Error parsing tavily response: {e}")
            items = []

        research_items = [i for i in items if i and i.strip()][:5]
        if research_items:
            logs.append(f"Research parsing succeeded: {len(research_items)} items.")
        else:
            logs.append("Research parsing returned no items.")
    else:
        logs.append("No response from Tavily SDK/wrapper.")

    # Fallback to samples.json if empty
    if not research_items:
        logs.append("Attempting to use samples.json fallback.")
        sample = sample_for_topic(topic)
        if sample and sample.get("pipeline"):
            sample_research = sample.get("research") or sample["pipeline"].get("ResearchAgent", "")
            if sample_research:
                logs.append("Using sample research from samples.json.")
                return sample_research, logs

    research_text = "\n\n".join(research_items) if research_items else ""
    return research_text, logs

# ---- Writer agent ----
def run_writer(topic: str, research_text: str, length: str, tone: str, logs: List[str]) -> Tuple[str, List[str]]:
    """Generate a draft using Gemini (or fallback if exhausted)."""
    # If DEMO_MODE and sample exists, use sample
    if DEMO_MODE:
        sample = sample_for_topic(topic)
        if sample and "pipeline" in sample and "WriterAgent" in sample["pipeline"]:
            logs.append("DEMO_MODE: using sample writer output.")
            return sanitize_text_output(sample["pipeline"]["WriterAgent"], topic=topic), logs

    # Ask Gemini to return plain Markdown for clean rendering
    prompt = f"Write a {length} {tone} {topic} as plain Markdown.\n\n"
    if research_text:
        prompt += "Use these research notes to support the content:\n" + research_text + "\n\n"
    prompt += "Return ONLY plain Markdown with: an H1 title, a short intro, 3-5 short paragraphs, and a short conclusion. Do NOT return JSON or code blocks."

    out = llm.run(prompt)
    if out == "GEMINI_RESOURCE_EXHAUSTED":
        logs.append("Gemini unavailable for writer -> using offline fallback summary.")
        return fallback_summary(topic, {"signal":"Unknown","confidence":0.0,"reason":"Gemini unavailable for writer"}), logs
    out = sanitize_text_output(out, topic=topic)
    logs.append("Writer produced content.")
    return out, logs

# ---- Editor agent ----
def run_editor(draft: str, logs: List[str]) -> Tuple[str, List[str]]:
    """Polish the draft using Gemini (or return draft if unavailable)."""
    if not draft:
        return "", logs
    prompt = "Polish and improve the clarity, grammar, and flow of the following content. Keep meaning unchanged:\n\n" + draft
    out = llm.run(prompt)
    if out == "GEMINI_RESOURCE_EXHAUSTED":
        logs.append("Gemini unavailable for editor -> returning original draft.")
        return draft, logs
    out = sanitize_text_output(out)
    logs.append("Editor polished the draft.")
    return out, logs

# ---- SEO agent ----
def run_seo(topic: str, research_text: str, draft: str, logs: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Produce SEO keywords and a short meta description.
    If the draft is the offline fallback summary, produce safe keywords (from topic or static).
    """
    # Detect if draft is the offline fallback summary by looking for the marker phrase we use
    fallback_marker = "Note: Cloud LLM temporarily unavailable"
    if draft and fallback_marker in draft:
        logs.append("Detected offline fallback draft; generating safe demo SEO keywords.")
        # create safe keywords from topic words (or default list)
        if topic:
            words = [w.strip(".,!?()[]\"'").lower() for w in topic.split() if len(w) > 3]
            keywords = ", ".join(dict.fromkeys(words))[:200]  # unique, simple
        else:
            keywords = "ai, demo, sample, content, offline"
        meta = (topic or "Demo content").strip()[:150]
        return {"keywords": keywords, "meta": meta}, logs

    # DEMO_MODE sample
    if DEMO_MODE:
        sample = sample_for_topic(topic)
        if sample and "pipeline" in sample and "SEOAgent" in sample["pipeline"]:
            logs.append("DEMO_MODE: using sample SEO output.")
            return {"keywords": sample["pipeline"]["SEOAgent"], "meta": sample.get("meta", "")}, logs

    # Try LLM to extract keywords and meta
    prompt = f"From the research and draft below, list 8 concise SEO keywords (comma separated) and a one-line meta description.\n\nResearch:\n{research_text}\n\nDraft:\n{draft}\n\nFormat:\nKeywords: keyword1, keyword2, ...\nMeta: one-sentence meta description"
    out = llm.run(prompt, max_retries=1)
    if out == "GEMINI_RESOURCE_EXHAUSTED":
        logs.append("Gemini unavailable for SEO -> using local keyword extraction.")
        words = [w.strip(".,!?()[]\"'").lower() for w in draft.split() if len(w) > 3]
        stop = set(["the","and","for","with","that","this","from","have","will","are","was","were","you","your"])
        freq = {}
        for w in words:
            if w in stop or w.isdigit():
                continue
            freq[w] = freq.get(w, 0) + 1
        keywords = ", ".join(sorted(freq, key=freq.get, reverse=True)[:8])
        meta = (draft.strip().split("\n")[0])[:150] if draft else ""
        return {"keywords": keywords, "meta": meta}, logs

    out_text = sanitize_text_output(out)
    # parse for keywords/meta
    keywords = ""
    meta = ""
    for line in out_text.splitlines():
        if line.lower().startswith("keywords"):
            parts = line.split(":", 1)
            keywords = parts[1].strip() if len(parts) > 1 else keywords
        if line.lower().startswith("meta"):
            parts = line.split(":", 1)
            meta = parts[1].strip() if len(parts) > 1 else meta

    # fallback if not parsed
    if not keywords:
        # first line may be comma-separated keywords
        first_line = out_text.strip().splitlines()[0] if out_text.strip().splitlines() else ""
        if "," in first_line:
            keywords = first_line.strip()
        else:
            # local extraction fallback
            words = [w.strip(".,!?()[]\"'").lower() for w in draft.split() if len(w) > 3]
            freq = {}
            stop = set(["the","and","for","with","that","this","from"])
            for w in words:
                if w in stop or w.isdigit(): continue
                freq[w] = freq.get(w, 0) + 1
            keywords = ", ".join(sorted(freq, key=freq.get, reverse=True)[:8])
    if not meta:
        meta = (draft.strip().split("\n")[0])[:150] if draft else ""
    logs.append("SEO produced keywords/meta.")
    return {"keywords": keywords, "meta": meta}, logs

# ---- CrewAI agent block (attempt to run with crew if available) ----
def run_crew_pipeline(topic: str, content_type: str, length: str, tone: str, use_research: bool, logs: List[str]) -> Dict[str, Any]:
    """
    Attempt to run a CrewAI crew to perform the pipeline.
    Returns the same dict shape as create_content expects. On any failure, returns None.
    """
    if not crew_available:
        logs.append("CrewAI not available in environment; skipping crew run.")
        return None

    try:
        logs.append("Attempting to run CrewAI pipeline...")

        # Build simple Agent definitions that reference our LLM/Tavily wrappers via 'tool' names
        # Note: different CrewAI versions may expect different inputs; we attempt a commonly supported pattern.
        researcher = Agent(
            name="Researcher",
            role="Find recent, trustworthy facts and sources",
            # use tavily tool if present; otherwise no tools
            tools=[tavily_tool] if tavily_tool else [],
            instructions="Search the web for recent facts, return a short JSON: {summary, highlights, urls}."
        )
        writer = Agent(
            name="Writer",
            role="Write a structured content piece using research",
            tools=[],  # writer will internally call Gemini via our llm in hooks below
            instructions="Write the requested content using research. Output Title, Intro, Body, Conclusion."
        )
        editor = Agent(
            name="Editor",
            role="Edit and polish the draft",
            tools=[],
            instructions="Improve clarity, grammar and SEO-readiness. Return edited draft and changelog."
        )
        seo = Agent(
            name="SEO",
            role="Produce keywords and meta",
            tools=[],
            instructions="From the final draft, produce 8 keywords and one-line meta."
        )

        # Create Tasks - mapping the flow Research -> Write -> Edit -> SEO
        t1 = Task(name="research", agent=researcher, output="research_summary")
        t2 = Task(name="write", agent=writer, input_from="research_summary", output="draft")
        t3 = Task(name="edit", agent=editor, input_from="draft", output="edited")
        t4 = Task(name="seo", agent=seo, input_from="edited", output="final_meta")

        crew = Crew(agents=[researcher, writer, editor, seo], tasks=[t1, t2, t3, t4], process=Process.sequential)

        # kickoff - many crewai versions accept inputs mapping; we pass topic / flags
        inputs = {"topic": topic, "content_type": content_type, "length": length, "tone": tone, "use_research": use_research}
        run_result = crew.kickoff(inputs=inputs)

        # Attempt to extract structured outputs - run_result structure varies between versions
        # We'll try several common shapes and normalize
        logs.append("Crew kickoff done, attempting to normalize crew output.")
        pipeline = {}
        research_text = ""
        final_output_text = ""
        seo_keywords = ""

        # If run_result is dict-like with 'outputs' or 'task_outputs'
        if isinstance(run_result, dict):
            # Look for known keys
            if "outputs" in run_result and isinstance(run_result["outputs"], dict):
                outputs = run_result["outputs"]
                research_text = outputs.get("research_summary") or outputs.get("research") or ""
                pipeline["ResearchAgent"] = research_text or "No research available"
                pipeline["WriterAgent"] = outputs.get("draft") or outputs.get("Writer") or ""
                pipeline["EditorAgent"] = outputs.get("edited") or outputs.get("Editor") or ""
                seo_keywords = outputs.get("final_meta") or outputs.get("seo") or ""
            elif "task_outputs" in run_result and isinstance(run_result["task_outputs"], dict):
                tasks = run_result["task_outputs"]
                research_text = tasks.get("research", "")
                pipeline["ResearchAgent"] = research_text or "No research available"
                pipeline["WriterAgent"] = tasks.get("write", "")
                pipeline["EditorAgent"] = tasks.get("edit", "")
                seo_keywords = tasks.get("seo", "")
            else:
                # fallback: stringify run_result
                pipeline["ResearchAgent"] = str(run_result.get("research_summary", "")) if run_result.get("research_summary") else "No research available"
                pipeline["WriterAgent"] = str(run_result.get("draft", "")) if run_result.get("draft") else ""
                pipeline["EditorAgent"] = str(run_result.get("edited", "")) if run_result.get("edited") else ""
                seo_keywords = str(run_result.get("final_meta", "")) if run_result.get("final_meta") else ""
        else:
            # if run_result is not dict, try str conversion
            pipeline["ResearchAgent"] = "No research available"
            pipeline["WriterAgent"] = str(run_result)
            pipeline["EditorAgent"] = str(run_result)
            seo_keywords = ""

        # Derive final output text from editor or writer
        final_output_text = pipeline.get("EditorAgent") or pipeline.get("WriterAgent") or fallback_summary(topic, {"signal":"CrewAI","confidence":0.0,"reason":"No content"})

        # Sanitize outputs
        sanitized_pipeline = {}
        for k, v in pipeline.items():
            if k == "SEOAgent":
                sanitized_pipeline[k] = v
            else:
                sanitized_pipeline[k] = sanitize_text_output(v, topic=topic)

        logs.append("CrewAI run completed and normalized.")
        return {
            "success": True,
            "final_output": sanitize_text_output(final_output_text, topic=topic),
            "pipeline": sanitized_pipeline,
            "research": research_text,
            "logs": logs
        }

    except Exception as e:
        logs.append(f"CrewAI pipeline failed: {e}")
        # On any failure, return None so fallback pipeline runs
        return None


# ---- Main create_content used by Streamlit ----
def create_content(topic: str,
                   content_type: str = "Blog",
                   length: str = "Short",
                   tone: str = "Informal",
                   use_research: bool = True) -> Dict[str, Any]:
    """
    Runs the sequential multi-agent pipeline and returns a consistent dict:
    {
      "success": bool,
      "final_output": str,
      "pipeline": { "ResearchAgent":..., "WriterAgent":..., "EditorAgent":..., "SEOAgent":... },
      "research": str,
      "logs": [...]
    }
    """
    logs: List[str] = []
    pipeline: Dict[str, Any] = {}

    # 0) If DEMO_MODE and sample exists -> directly return sample
    if DEMO_MODE:
        sample = sample_for_topic(topic)
        if sample:
            logs.append("DEMO_MODE: returning sample output.")
            return {
                "success": True,
                "final_output": sanitize_text_output(sample.get("final_output", ""), topic=topic),
                "pipeline": {k: sanitize_text_output(v, topic=topic) for k,v in sample.get("pipeline", {}).items()},
                "research": sample.get("research", ""),
                "logs": logs
            }

    # Try CrewAI first (if available)
    if crew_available:
        crew_result = run_crew_pipeline(topic=topic, content_type=content_type, length=length, tone=tone, use_research=use_research, logs=logs)
        if crew_result:
            logs.append("Using CrewAI pipeline result.")
            return crew_result
        else:
            logs.append("CrewAI attempt failed or returned no result; falling back to local pipeline.")

    # 1) Research
    research_text = ""
    if use_research:
        research_text, logs = run_research(topic, logs)
    else:
        logs.append("use_research=False: skipping research.")

    pipeline["ResearchAgent"] = research_text if research_text else "No research available"

    # 2) Writer
    writer_out, logs = run_writer(topic, research_text, length, tone, logs)
    pipeline["WriterAgent"] = writer_out

    # 3) Editor
    editor_out, logs = run_editor(writer_out, logs)
    pipeline["EditorAgent"] = editor_out

    # If editor_out contains fallback marker (Gemini unavailable) but we have research, build a safe local draft
    fallback_marker = "Note: Cloud LLM temporarily unavailable"
    if editor_out and fallback_marker in editor_out and research_text:
        logs.append("Gemini unavailable — building local draft from research_text as fallback.")
        # Build a simple draft from research items (research_text is blocks separated by double newlines)
        blocks = [b.strip() for b in research_text.split("\n\n") if b.strip()]
        # Compose a local draft: title, intro, bullets from top 3 research items, conclusion
        title = f"{topic} — Research Summary"
        intro = f"This is an automatically assembled research summary for the topic: {topic}."
        body_parts = []
        for i, b in enumerate(blocks[:4], start=1):
            # make small paragraph for each block
            body_parts.append(f"• Research item {i}:\n{b}")
        conclusion = "Summary generated from live research results (Tavily)."
        local_draft = f"# {title}\n\n{intro}\n\n" + "\n\n".join(body_parts) + f"\n\n{conclusion}"
        editor_out = local_draft
        pipeline["EditorAgent"] = editor_out

    # 4) SEO
    seo_out, logs = run_seo(topic, research_text, editor_out, logs)
    pipeline["SEOAgent"] = seo_out.get("keywords", "")

    # final output take from editor_out (polished)
    final_output_text = editor_out or writer_out or fallback_summary(topic, {"signal":"Unknown","confidence":0.0,"reason":"No content produced"})

    # sanitize final and pipeline items
    final_output_text = sanitize_text_output(final_output_text, topic=topic)

    # ---- IMPORTANT: preserve SEO keywords string, sanitize everything else ----
    sanitized_pipeline = {}
    for k, v in pipeline.items():
        if k == "SEOAgent":
            # leave SEO keywords as-is (string)
            sanitized_pipeline[k] = v
        else:
            try:
                sanitized_pipeline[k] = sanitize_text_output(v, topic=topic)
            except Exception:
                sanitized_pipeline[k] = "No output available."

    return {
        "success": True,
        "final_output": final_output_text,
        "pipeline": sanitized_pipeline,
        "research": research_text,
        "logs": logs
    }
