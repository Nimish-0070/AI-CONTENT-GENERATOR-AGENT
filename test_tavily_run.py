# test_tavily_run.py
from dotenv import load_dotenv
load_dotenv()

from agent import run_research  # uses the run_research in your agent.py
topic = "latest advances in electric vehicle batteries 2025"

res, logs = run_research(topic, [])
print("=== LOGS ===")
for l in logs:
    print("-", l)
print("\n=== RESEARCH (first 2000 chars) ===\n")
print(res[:2000] or "(no research returned)")
