"""GET /api/health — Vercel serverless function (stdlib only)."""
import os
import sys
from http.server import BaseHTTPRequestHandler

# Each api/*.py is bundled as its own lambda; add the shared api/ dir to sys.path so the
# sibling `_engine` module resolves there (Vercel does not add it automatically).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import load_model, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        M = load_model()
        send_json(self, 200, {"status": "ok" if M else "no_model",
                              "teams": len(M["squads"]) if M else 0})

    def log_message(self, *args):
        pass
