"""ds41api.py - where the checks find the ds41 server: DS41_URL (http://127.0.0.1:18080; larry: http://172.23.0.1:18080),
the key DS41_API_KEY or the VLLM_API_KEY=... line of DS41_KEY_FILE (neither: no Authorization header), DS41_HOME (the
directory these scripts are installed in)."""
import json
import os
import urllib.request

D = os.environ.get('DS41_HOME') or os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get('DS41_URL', 'http://127.0.0.1:18080').rstrip('/') + '/v1/chat/completions'


def key():
    k = os.environ.get('DS41_API_KEY', '')
    f = os.environ.get('DS41_KEY_FILE', '')
    if not k and f:
        k = [ln.split('=', 1)[1].strip() for ln in open(f) if ln.startswith('VLLM_API_KEY=')][0]
    return k


def chat(body, timeout=1800):
    """POST one chat completion; the response as a dict"""
    h = {'Content-Type': 'application/json'}
    if key():
        h['Authorization'] = 'Bearer ' + key()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)
