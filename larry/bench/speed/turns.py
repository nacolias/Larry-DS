#!/usr/bin/env python3
"""turns.py ENGINE_LOG [SINCE_LINE] - per request of the serve engine's log: prompt (new / reused / batch or passes, s,
experts cpu), decode (tokens, s, tok/s, DSpark passes, tokens a pass, ms a pass, CPU-pool experts a pass, hit a pass)."""
import re, sys
lines = open(sys.argv[1], errors='replace').read().splitlines()
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rp = re.compile(r'^serve: prompt (\d+) tokens \((\d+) reused, \w+\) in ([\d.]+) s = [\d.]+ tok/s \(experts (\d+) hit / (\d+) cpu / (\d+) pcie')
rph = re.compile(r'^serve: prompt phases: (\d+) new = (\d+) batch .*DSpark tail (\d+) (from the batch|as passes) \(cache open ([\d.]+) s, seed ([\d.]+) s\)')
rd = re.compile(r'^serve: (\d+) tokens in ([\d.]+) s = [\d.]+ tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu')
rs = re.compile(r'^serve: DSpark (\d+) passes, ([\d.]+) tokens/pass')
cur = {}
print('%7s %6s %6s %5s %6s %6s | %6s %7s %6s %6s %5s %6s %6s %6s' % ('prompt', 'new', 'batch', 'tail', 'p_s', 'p_cpu', 'dec', 'd_s', 'tok/s', 'passes', 'tpp', 'ms/p', 'cpu/p', 'hit/p'))
for ln in lines[start:]:
    m = rph.match(ln)
    if m:
        cur = {'new': int(m.group(1)), 'batch': int(m.group(2)), 'tail': int(m.group(3)), 'mode': m.group(4)[0], 'open': float(m.group(5)), 'seed': float(m.group(6))}
        continue
    m = rp.match(ln)
    if m:
        cur.update(prompt=int(m.group(1)), reused=int(m.group(2)), ps=float(m.group(3)), pcpu=int(m.group(5)))
        continue
    m = rd.match(ln)
    if m:
        cur.update(dec=int(m.group(1)), ds=float(m.group(2)), hit=int(m.group(4)), cpu=int(m.group(5)))
        continue
    m = rs.match(ln)
    if m and 'dec' in cur:
        p = int(m.group(1))
        c = cur
        print('%7d %6d %6d %4d%s %6.2f %6d | %6d %7.2f %6.1f %6d %5.2f %6.1f %6.1f %6.1f' % (
            c.get('prompt', 0), c.get('new', 0), c.get('batch', 0), c.get('tail', 0), c.get('mode', '?'), c.get('ps', 0), c.get('pcpu', 0),
            c['dec'], c['ds'], c['dec'] / max(c['ds'], 1e-9), p, c['dec'] / max(p, 1), 1e3 * c['ds'] / max(p, 1), c['cpu'] / max(p, 1), c['hit'] / max(p, 1)))
        cur = {}
