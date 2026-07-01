"""GET /api/health — Vercel serverless function (stdlib only)."""
from http.server import BaseHTTPRequestHandler

from _engine import load_model, send_json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        M = load_model()
        send_json(self, 200, {"status": "ok" if M else "no_model",
                              "teams": len(M["squads"]) if M else 0})

    def log_message(self, *args):
        pass
