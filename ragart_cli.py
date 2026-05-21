"""ragart — console entry point.

Installed as the `ragart` command via pyproject.toml's [project.scripts].
Starts the RagArt web server and (by default) opens it in a browser:

    ragart                 # start on the configured host/port
    ragart --port 8080     # override the port
    ragart --no-browser    # don't open a browser (servers, Docker)
    ragart --debug         # Flask debug mode (auto-reload)
"""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragart",
        description="RagArt — Turkish Retrieval-Augmented Generation platform.",
    )
    parser.add_argument("--host", default=None,
                        help="Bind host (default: from settings, 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Bind port (default: from settings, 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window")
    parser.add_argument("--debug", action="store_true",
                        help="Flask debug mode (auto-reload)")
    args = parser.parse_args()

    # Imported here so `ragart --help` is instant (no model/registry load).
    from config.settings import settings
    from app import app

    host = args.host or settings.HOST
    port = args.port or settings.PORT
    url = f"http://localhost:{port}"

    print(f"\n  RagArt → {url}")
    print("  (durdurmak için Ctrl+C)\n")

    if not args.no_browser:
        # Open the browser shortly after the server has had time to bind.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # debug defaults off: no reloader → no double browser-open, clean run.
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
