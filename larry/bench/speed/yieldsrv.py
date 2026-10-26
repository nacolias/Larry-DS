#!/usr/bin/env python3
"""yieldsrv.py FLAG [--port 18080] [--max-s 5400] - stands in for the ds41 service on its port while an A/B window
holds it stopped: any request gets HTTP 503 (an OpenAI-style error, Retry-After 180) and creates FLAG, which tells the
window to stop its tests and bring the service back.  Exits on SIGTERM or after --max-s seconds (never outlives a
window that died).  Run with /opt/ai/dsv41/venv/bin/python -I."""
import argparse
import http.server
import json
import os
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument('flag')
ap.add_argument('--port', type=int, default=18080)
ap.add_argument('--max-s', type=int, default=5400)
a = ap.parse_args()
MSG = json.dumps({'error': {'message': 'larry is running a short benchmark - the model is back in about 3 minutes',
                            'type': 'service_unavailable', 'code': 503}}).encode()


class H(http.server.BaseHTTPRequestHandler):
    def answer(self):
        if not os.path.exists(a.flag):
            with open(a.flag, 'w') as f:
                f.write('%s %s %s\n' % (time.strftime('%F %T'), self.command, self.path))
        try:
            n = int(self.headers.get('Content-Length') or 0)
            if 0 < n < 1 << 24:
                self.rfile.read(n)
        except Exception:
            pass
        self.send_response(503)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Retry-After', '180')
        self.send_header('Content-Length', str(len(MSG)))
        self.end_headers()
        self.wfile.write(MSG)

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = answer

    def log_message(self, fmt, *args):
        print('%s %s' % (time.strftime('%T'), fmt % args), flush=True)


srv = http.server.ThreadingHTTPServer(('0.0.0.0', a.port), H)
t = threading.Timer(a.max_s, srv.shutdown)
t.daemon = True
t.start()
print('%s: standing in on :%d (flag %s, at most %d s)' % (time.strftime('%T'), a.port, a.flag, a.max_s), flush=True)
srv.serve_forever()
