"""ragart — console entry point.

Installed as the `ragart` command via pyproject.toml's [project.scripts].
Starts the RagArt web server and (by default) opens it in a browser:

    ragart                 # dev server on the configured host/port
    ragart --port 8080     # override the port
    ragart --no-browser    # don't open a browser (servers, Docker)
    ragart --debug         # Flask debug mode (auto-reload)
    ragart --production    # serve with waitress (no dev-server warning)

For heavier production deploys (multiple SSE clients) prefer gunicorn +
gevent — see gunicorn.conf.py / Procfile / Dockerfile.
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
    parser.add_argument("--production", action="store_true",
                        help="Serve with the waitress production WSGI server "
                             "(cross-platform; no Flask dev-server warning)")
    args = parser.parse_args()

    # Imported here so `ragart --help` is instant (no model/registry load).
    from config.settings import settings
    from app import app

    host = args.host or settings.HOST
    port = args.port or settings.PORT
    url = f"http://localhost:{port}"

    mode = "production (waitress)" if args.production else "dev"
    print(f"\n  RagArt [{mode}] → {url}")
    print("  (durdurmak için Ctrl+C)\n")

    if not args.no_browser:
        # Open the browser shortly after the server has had time to bind.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    if args.production:
        try:
            from waitress import serve
        except ImportError:
            raise SystemExit(
                "waitress is not installed. Install the production extras:\n"
                "    pip install 'ragart[prod]'   (or: pip install waitress)"
            )
        # threads: a small pool is plenty for a single-instance showcase;
        # SSE responses stream as the generator yields.
        serve(app, host=host, port=port, threads=8)
    else:
        # debug defaults off: no reloader → no double browser-open, clean run.
        app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
