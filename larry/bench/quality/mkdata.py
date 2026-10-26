#!/usr/bin/env python3
"""mkdata.py ROWS OUT - MathArena/aime_2026 (datasets-server rows JSON) -> DeepInfer's JSONL records
(task_id aime_2026_<problem_idx - 1>, question, answer as an integer string)."""
import json, sys
rows = json.load(open(sys.argv[1]))['rows']
with open(sys.argv[2], 'w') as f:
    for r in sorted(rows, key=lambda r: r['row']['problem_idx']):
        r = r['row']
        f.write(json.dumps({'task_id': 'aime_2026_%d' % (r['problem_idx'] - 1), 'question': r['problem'],
                            'answer': str(r['answer'])}, ensure_ascii=False) + '\n')
print(len(rows), 'problems ->', sys.argv[2])
