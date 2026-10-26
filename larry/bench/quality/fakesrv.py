#!/usr/bin/env python3
"""fakesrv.py PORT - a stand-in for the ds41 server to test the eval clients' yield: /v1/chat/completions streams a
reasoning chunk every 0.5 s for 60 s (then an answer), /metrics reports one request queued from 12 s after start."""
import http.server, json, sys, time
T0 = time.time()
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({'live': {'queued': 1 if time.time() - T0 > 12 else 0}}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        try:
            for i in range(120):
                ev = {'choices': [{'delta': {'reasoning_content': 'step %d ' % i}}]}
                self.wfile.write(b'data: ' + json.dumps(ev).encode() + b'\n\n'); self.wfile.flush(); time.sleep(0.5)
            for ev in ({'choices': [{'delta': {'content': '\\boxed{277}'}, 'finish_reason': 'stop'}]},
                       {'choices': [], 'usage': {'completion_tokens': 121}}):
                self.wfile.write(b'data: ' + json.dumps(ev).encode() + b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            print('fake: stream completed', flush=True)
        except (BrokenPipeError, ConnectionResetError):
            print('fake: client closed after %.1f s' % (time.time() - T0), flush=True)
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
