"""Dashboard server: serves static files from project root + 3 JSON endpoints.

Usage:
    uv run python scripts/dashboard_server.py
    # then open http://localhost:8000/dashboard/

Endpoints:
    /api/runs              → list of {exp_id, timestamp} for every run dir
    /api/retrieval?run=... → retrieval.yaml as JSON
    /api/meta?run=...      → meta.yaml as JSON (or null if not yet written)
    /api/conversation?run=... → conversation.md raw text

(plus static file serving from project root)
"""
import http.server
import json
import socketserver
import sys
import urllib.parse
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
PORT = 8000


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/runs":
            self._send_json(self._list_runs())
            return
        if parsed.path == "/api/retrieval":
            self._send_yaml_as_json(self._run_file_path(parsed, "retrieval.yaml"))
            return
        if parsed.path == "/api/meta":
            self._send_yaml_as_json(self._run_file_path(parsed, "meta.yaml"))
            return
        if parsed.path == "/api/conversation":
            self._send_text(self._run_file_path(parsed, "conversation.md"))
            return
        super().do_GET()

    def _list_runs(self):
        if not RUNS_DIR.exists():
            return []
        runs = []
        for exp_dir in sorted(RUNS_DIR.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            for ts_dir in sorted(exp_dir.iterdir(), reverse=True):
                if ts_dir.is_dir() and not ts_dir.name.startswith("."):
                    runs.append({"exp_id": exp_dir.name, "timestamp": ts_dir.name})
        # Latest overall first: sort by timestamp desc
        runs.sort(key=lambda r: r["timestamp"], reverse=True)
        return runs

    def _run_file_path(self, parsed, filename: str) -> Path | None:
        qs = urllib.parse.parse_qs(parsed.query)
        run_param = qs.get("run", [None])[0]
        if not run_param:
            return None
        # Expect run=<exp_id>/<timestamp>
        return RUNS_DIR / run_param / filename

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_yaml_as_json(self, path: Path | None):
        if not path or not path.exists():
            self._send_json(None)
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self._send_json(data)

    def _send_text(self, path: Path | None):
        if not path or not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        body = path.read_text(encoding="utf-8").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress noisy access logs; only show errors
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard serving on http://localhost:{PORT}/dashboard/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
