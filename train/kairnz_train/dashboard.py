"""Web dashboard server for the AlphaZero training loop.

Serves a static HTML page at GET / and JSON endpoints at GET /api/status
and GET /api/metrics.  All three endpoints read from the work directory
produced by the training orchestrator.

Run as:
    python -m kairnz_train.dashboard --work /path/to/work --port 8080
Then reach it via an SSH tunnel on the remote machine.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kairnz_train.orchestrate import load_metrics, load_status

# Defaults for the CLI
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# File names inside the work directory
STATUS_FILENAME = "status.json"
METRICS_FILENAME = "metrics.jsonl"

# Placeholder served when the static HTML has not been built yet (Task 10)
_HTML_PLACEHOLDER = b"<html><body><p>Dashboard loading...</p></body></html>"


def status_payload(work: Path) -> dict:
    """Returns the current training status dict from work/status.json.

    Delegates to load_status, which returns an empty dict when the file is
    missing or unreadable.
    """
    return load_status(work / STATUS_FILENAME)


def metrics_payload(work: Path) -> list[dict]:
    """Returns the training metrics list from work/metrics.jsonl.

    Delegates to load_metrics, which returns an empty list when the file is
    missing.
    """
    return load_metrics(work / METRICS_FILENAME)


def _make_handler(work: Path, html_path: Path):
    """Returns a BaseHTTPRequestHandler subclass closed over work and html_path."""

    class _Handler(BaseHTTPRequestHandler):
        """HTTP request handler for the training dashboard."""

        def _send_json(self, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_404(self) -> None:
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 -- stdlib naming convention
            """Dispatch GET requests to the appropriate handler."""
            if self.path in ("/", "/index.html"):
                if html_path.exists():
                    self._send_html(html_path.read_bytes())
                else:
                    self._send_html(_HTML_PLACEHOLDER)
            elif self.path == "/api/status":
                self._send_json(status_payload(work))
            elif self.path == "/api/metrics":
                self._send_json(metrics_payload(work))
            else:
                self._send_404()

        def log_message(self, fmt: str, *args) -> None:  # noqa: ANN001
            """Suppress default access-log noise; errors still go to stderr."""

    return _Handler


def _build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m kairnz_train.dashboard",
        description="Serve the Kairnz training dashboard over HTTP.",
    )
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Work directory containing status.json and metrics.jsonl.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST}). Reach via SSH tunnel.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT}).",
    )
    return parser


def main() -> None:
    """Parse CLI args and start the dashboard server."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    work: Path = args.work
    html_path = Path(__file__).parent / "static" / "dashboard.html"

    handler_cls = _make_handler(work, html_path)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Dashboard at http://{args.host}:{args.port}/  (work={work})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
