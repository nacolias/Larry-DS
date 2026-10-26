#!/usr/bin/env python3
# abw-catfix.py [ABW_SH] - larry, 2026-10-25: abw.sh compared dmesg's CAT-error COUNT with the window start's - the ring
# buffer rotating old lines out (8 -> 6) read as "new CAT errors" and left the service down with the hold kept (abwT13,
# 03:22-03:31).  Now: the kernel journal's CAT errors since the window start (journalctl -k --since).  Idempotent.
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/opt/ai/dsv41/bench/speed/abw.sh"
s = open(P).read()
if "cat_new()" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, n=1):
    global s
    if s.count(old) != n:
        sys.exit("anchor x%d (want %d): %s" % (s.count(old), n, old[:80]))
    s = s.replace(old, new)


rep('cat_errs() { dmesg | grep -c -i "cat error"; }\n',
    'cat_errs() { dmesg | grep -c -i "cat error"; }\n'
    '# CAT errors logged since the window started (the kernel journal: dmesg\'s ring buffer drops old lines)\n'
    'cat_new() { journalctl -k --since "$WSTART" --no-pager 2>/dev/null | grep -c -i "cat error"; }\n')
rep('c0=$(cat_errs)\n', 'c0=$(cat_errs)\nWSTART=$(date \'+%F %T\')\n')
rep('''    c=$(cat_errs)
    if [ "$c" != "$c0" ]; then
      echo "NEW CAT ERRORS in dmesg ($c0 -> $c) after the probe: stopping, service NOT restarted, hold kept"''',
    '''    c=$(cat_new)
    if [ "$c" -gt 0 ]; then
      echo "NEW CAT ERRORS in the kernel log ($c since $WSTART) after the probe: stopping, service NOT restarted, hold kept"''')
rep('''  c=$(cat_errs)
  if [ "$c" != "$c0" ]; then
    echo "NEW CAT ERRORS in dmesg ($c0 -> $c) after $N: stopping, service NOT restarted, hold kept"''',
    '''  c=$(cat_new)
  if [ "$c" -gt 0 ]; then
    echo "NEW CAT ERRORS in the kernel log ($c since $WSTART) after $N: stopping, service NOT restarted, hold kept"''')
open(P, "w").write(s)
print("patched", P)
