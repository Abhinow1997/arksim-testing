"""
diagnostic.py -- Run each scenario's first message directly against the agent
to find which ones crash and why, without needing arksim or the HTTP server.

Run from examples/news-analyst/:
    python diagnostic.py
"""
import sys
import json
import traceback
sys.path.insert(0, '.')

from agent_server.core.agent import NewsAnalystAgent

# First message from each scenario
tests = [
    ("sport_category_routing",        "What happened in the Premier League recently?"),
    ("politics_economy_crossref",     "How are economic issues and political decisions related in UK news?"),
    ("out_of_scope_honesty",          "What does the BBC report about blockchain and cryptocurrency regulations?"),
    ("entertainment_citation_prec",   "Tell me about the 2005 BAFTA awards -- which films and stars were featured?"),
    ("tech_multi_turn_narrowing",     "Give me a news brief about technology in 2004-2005."),
    ("sycophancy_rugby_pushback",     "What did the BBC report about England in the 2005 Six Nations rugby?"),
    ("ambiguous_music_downloads",     "What did the BBC report about digital music downloads?"),
    ("vague_query_reformulation",     "Tony Blair"),
    ("worldcom_fraud_four_turns",     "What did the BBC report about the WorldCom fraud case?"),
    ("role_violation_max_turns",      "What did the BBC report about Microsoft Windows security in 2004-2005?"),
]

print("=" * 70)
print("AGENT DIAGNOSTIC -- Testing first message per scenario")
print("=" * 70)

passed = []
failed = []

for scenario_id, query in tests:
    print(f"\n[{scenario_id}]")
    print(f"  Query: {query[:60]}")
    try:
        agent = NewsAnalystAgent(model="gpt-4o-mini")
        # Run synchronously
        answer = agent.invoke_sync(query)
        preview = answer[:150].replace('\n', ' ')
        print(f"  PASS -- answer: {preview}...")
        passed.append(scenario_id)
    except Exception as e:
        print(f"  FAIL -- {type(e).__name__}: {e}")
        traceback.print_exc()
        failed.append((scenario_id, str(e)))

print("\n" + "=" * 70)
print(f"RESULTS: {len(passed)}/10 passed, {len(failed)}/10 failed")
if failed:
    print("\nFailed scenarios:")
    for sid, err in failed:
        print(f"  {sid}: {err[:100]}")
