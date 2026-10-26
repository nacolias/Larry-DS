#!/usr/bin/env python3
"""mkgpqa.py OUT ROWS.json ... - hendrydong/gpqa_diamond_mc (datasets-server rows JSON) -> DeepInfer's GPQA records as
its loader normalises them: the mirror's trailing "Please write your final answer in the form of ..." stripped, the
answer \\boxed{X} -> X, task_id gpqa_diamond_<row>, category = domain."""
import json, re, sys
TRAIL = re.compile(r"\n*\s*Please write your final answer in the form of.*$", re.I | re.S)
rows = []
for p in sys.argv[2:]:
    rows += json.load(open(p))['rows']
rows.sort(key=lambda r: r['row_idx'])
with open(sys.argv[1], 'w') as f:
    for r in rows:
        x = r['row']
        m = re.search(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", x['solution'])
        f.write(json.dumps({'task_id': 'gpqa_diamond_%d' % r['row_idx'], 'question': TRAIL.sub('', x['problem']).strip(),
                            'answer': m.group(1), 'category': x.get('domain', 'unknown')}, ensure_ascii=False) + '\n')
print(len(rows), 'questions ->', sys.argv[1])
