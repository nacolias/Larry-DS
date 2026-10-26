#!/usr/bin/env python3
"""tooltest.py - an OpenAI tool-call round trip against the ds41 server (ds41api.py)."""
import json, sys, time
import ds41api
def post(body):
    t = time.time()
    d = ds41api.chat(body, timeout=600)
    return d, time.time() - t
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}},
                         "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Oslo right now, in Celsius? Use the tool."}]
d, dt = post({"model": "larry", "messages": msgs, "tools": tools, "max_tokens": 600, "reasoning_effort": "none"})
m = d["choices"][0]["message"]
print(f"turn 1 ({dt:.1f} s): finish={d['choices'][0]['finish_reason']} tool_calls={json.dumps(m.get('tool_calls'))} content={m.get('content')!r}")
if not m.get("tool_calls"):
    sys.exit(1)
tc = m["tool_calls"][0]
msgs.append(m)
msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"city": "Oslo", "temp_c": 4.5, "conditions": "light rain"})})
d, dt = post({"model": "larry", "messages": msgs, "tools": tools, "max_tokens": 300, "reasoning_effort": "none"})
print(f"turn 2 ({dt:.1f} s): {d['choices'][0]['message'].get('content')!r}")
print("timings:", d.get("timings", {}).get("predicted_per_second"), "tok/s")
