"""GET /api/teams — Vercel serverless function (stdlib only)."""
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find sibling _engine.py in the lambda
from _engine import load_model, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        M = load_model()
        if not M:
            return send_json(self, 503, {"detail": "model.json not found"})
        send_json(self, 200, {"teams": sorted(M["squads"]), "meta": M.get("meta", {})})

    def log_message(self, *args):
        pass
