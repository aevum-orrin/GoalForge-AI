"""POST /api/predict — Vercel serverless function (stdlib only)."""
import json
from http.server import BaseHTTPRequestHandler

from _engine import load_model, predict_dict, send_json


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        M = load_model()
        if not M:
            return send_json(self, 503, {"detail": "model.json not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return send_json(self, 400, {"detail": "invalid JSON body"})
        h, a = req.get("home_team"), req.get("away_team")
        if h not in M["squads"] or a not in M["squads"]:
            return send_json(self, 404, {"detail": "unknown team"})
        if h == a:
            return send_json(self, 400, {"detail": "home_team and away_team must differ"})
        hx = req.get("home_xi") or M["squads"][h][:11]
        ax = req.get("away_xi") or M["squads"][a][:11]
        send_json(self, 200, predict_dict(M, h, a, hx, ax, bool(req.get("neutral", True))))

    def log_message(self, *args):
        pass
