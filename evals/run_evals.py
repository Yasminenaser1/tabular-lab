"""Run the assistant against a fixed set of questions and score the answers.

    python3 evals/run_evals.py

Each case checks: which tool the assistant chose (or none), words the answer
must contain, and words it must not. Checks are case-insensitive. Runs locally
against Ollama; nothing here needs an API key or the internet.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent

CASES = json.loads((Path(__file__).parent / "cases.json").read_text())


def check(case: dict, answer: str, tool_log: list) -> list:
    """Return a list of failure messages; empty means the case passed."""
    fails = []
    tools_used = [entry["tool"] for entry in tool_log]

    expected = case["expect_tool"]
    if expected is None:
        if tools_used:
            fails.append(f"expected no tool, but used {tools_used}")
    elif expected not in tools_used:
        fails.append(f"expected tool {expected!r}, used {tools_used or 'none'}")

    low = answer.lower()
    for word in case.get("must_include", []):
        if word.lower() not in low:
            fails.append(f"missing {word!r}")
    for word in case.get("must_exclude", []):
        if word.lower() in low:
            fails.append(f"should not contain {word!r}")
    return fails


def main():
    model = os.getenv("ASSISTANT_MODEL", agent.MODEL)
    print(f"Running {len(CASES)} cases against {model}\n")
    passed = 0
    total_time = 0.0
    for i, case in enumerate(CASES, 1):
        start = time.time()
        answer, tool_log = agent.ask([{"role": "user", "content": case["question"]}])
        elapsed = time.time() - start
        total_time += elapsed
        fails = check(case, answer, tool_log)
        if fails:
            print(f"{i}. FAIL ({elapsed:.0f}s)  {case['question']}")
            for f in fails:
                print(f"       - {f}")
            print(f"       answer: {answer[:150]}")
        else:
            passed += 1
            print(f"{i}. pass ({elapsed:.0f}s)  {case['question']}")
    print(f"\n{passed}/{len(CASES)} passed  |  {total_time:.0f}s total, {total_time / len(CASES):.0f}s per case")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
