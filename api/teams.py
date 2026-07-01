"""GET /api/teams — Vercel serverless function (stdlib only)."""
from http.server import BaseHTTPRequestHandler

from _engine import load_model, send_json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        M = load_model()
        if not M:
            return send_json(self, 503, {"detail": "model.json not found"})
        send_json(self, 200, {"teams": sorted(M["squads"]), "meta": M.get("meta", {})})

    def log_message(self, *args):
        pass
