#!/usr/bin/env python3
"""qv_fetch.py OUT.json - public texts for the quant-vs-original check (HF datasets-server rows API, no downloads of
whole datasets): English / Chinese Wikipedia articles, MATH-500 + GSM8K worked solutions, SWE-agent trajectories (as
chat messages).  Code sequences are added on larry from local files (qv_build.py)."""
import json, random, sys, urllib.parse, urllib.request

API = "https://datasets-server.huggingface.co/rows"


def rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for _ in range(4):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))["rows"]]
        except Exception as e:  # rate limits / transient errors
            err = e
    raise err


def wiki(config, total, want, min_chars, rng):
    got, seen = [], set()
    while len(got) < want:
        for r in rows("wikimedia/wikipedia", config, "train", rng.randrange(total - 100), 100):
            if len(r["text"]) >= min_chars and r["title"] not in seen and len(got) < want:
                seen.add(r["title"])
                got.append(r["title"] + "\n\n" + r["text"])
    return got


def main():
    rng = random.Random(20261022)
    out = {"en": wiki("20231101.en", 6407814, 4, 12000, rng), "zh": wiki("20231101.zh", 1384748, 3, 6000, rng)}
    math = rows("HuggingFaceH4/MATH-500", "default", "test", 0, 100) + rows("HuggingFaceH4/MATH-500", "default", "test", 100, 100)
    rng.shuffle(math)
    docs, cur = [], ""
    for r in math:
        cur += f"Problem: {r['problem']}\n\nSolution: {r['solution']}\n\n"
        if len(cur) > 12000:
            docs.append(cur); cur = ""
        if len(docs) == 2:
            break
    gsm = rows("openai/gsm8k", "main", "test", rng.randrange(1200), 60)
    docs.append("".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in gsm))
    out["math"] = docs
    agent, repos = [], set()
    while len(agent) < 4:
        for r in rows("nebius/SWE-agent-trajectories", "default", "train", rng.randrange(80000), 20):
            repo = r["instance_id"].rsplit("-", 1)[0]
            if repo in repos or len(agent) >= 4:
                continue
            traj = r["trajectory"]
            if isinstance(traj, str):
                import ast
                traj = ast.literal_eval(traj)
            msgs = []
            for m in traj:
                role, text = m.get("role"), m.get("content") or m.get("text") or ""
                if role == "system" or not isinstance(text, str) or not text.strip():
                    continue
                msgs.append({"role": "assistant" if role in ("assistant", "ai") else "user", "content": text})
            if len(msgs) >= 6 and sum(len(m["content"]) for m in msgs) > 12000:
                repos.add(repo)
                agent.append({"id": r["instance_id"], "messages": msgs})
    out["agent"] = agent
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
    print({k: [len(v) if isinstance(v, str) else sum(len(m["content"]) for m in v["messages"]) for v in vs] for k, vs in out.items()})


if __name__ == "__main__":
    main()
