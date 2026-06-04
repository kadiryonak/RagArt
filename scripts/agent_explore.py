"""Agent-based exploratory UI crawler for RagArt (Playwright, Python).

Walks the web UI, clicks every interactive element it hasn't seen before,
watches for JS/console/network errors, screenshots each step, and RECORDS a
signature for every element it exercised so subsequent runs SKIP what's
already been tested ("agent gezsin, her şeyi denesin, kaydetsin, tekrarlamasın").

State persists in tests/e2e/.explore-state.json; a per-run report lands in
tests/e2e/explore-report.json.

Setup (one-time):
    pip install playwright
    playwright install chromium

Run (RagArt must be serving on the URL below):
    python scripts/agent_explore.py --url http://localhost:5000
    python scripts/agent_explore.py --reset      # forget visited, explore fresh
    python scripts/agent_explore.py --headed      # watch it click around

This is a *fuzz/smoke* explorer, complementary to the deterministic specs in
tests/e2e/ragart.spec.js and the Python integration suite. It is intentionally
defensive: detached nodes, navigations and dialogs are handled, and the run
is capped so it always terminates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

STATE_PATH = Path("tests/e2e/.explore-state.json")
REPORT_PATH = Path("tests/e2e/explore-report.json")
SHOTS_DIR = Path("tests/e2e/explore-shots")

# Selectors for "things a user can interact with".
INTERACTIVE = "button, a[href], [role=button], [role=tab], select, summary, input[type=checkbox]"


def _sig(role: str, text: str, ident: str) -> str:
    """Stable signature for an element so we don't re-test it across runs."""
    raw = f"{role}|{(text or '').strip()[:60]}|{ident or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _load_state() -> set:
    if STATE_PATH.exists():
        try:
            return set(json.loads(STATE_PATH.read_text("utf-8")).get("visited", []))
        except Exception:
            return set()
    return set()


def _save_state(visited: set) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"visited": sorted(visited)}, indent=2), "utf-8")


def explore(url: str, *, headed: bool, max_steps: int) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright is not installed. Run:\n"
            "    pip install playwright && playwright install chromium"
        )

    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    visited = _load_state()
    report = {"url": url, "started": time.time(), "steps": [], "errors": [], "new_elements": 0}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        page = browser.new_page()

        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
        # Auto-dismiss native dialogs so the crawl never blocks.
        page.on("dialog", lambda d: d.dismiss())

        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(800)

        steps = 0
        while steps < max_steps:
            handles = page.query_selector_all(INTERACTIVE)
            # Find the first not-yet-visited, currently-visible element.
            target = None
            tsig = None
            for h in handles:
                try:
                    if not h.is_visible():
                        continue
                    role = (h.get_attribute("role") or h.evaluate("e=>e.tagName") or "").lower()
                    text = h.inner_text() if h.evaluate("e=>e.tagName") != "SELECT" else "select"
                    ident = h.get_attribute("id") or h.get_attribute("name") or ""
                    s = _sig(role, text, ident)
                    if s not in visited:
                        target, tsig = h, s
                        break
                except Exception:
                    continue

            if target is None:
                break  # nothing new to try → exploration converged

            before_errs = len(console_errors)
            label = ""
            try:
                label = (target.inner_text() or target.get_attribute("id") or "?")[:50]
                target.scroll_into_view_if_needed(timeout=2000)
                target.click(timeout=3000)
                page.wait_for_timeout(500)
            except Exception as e:
                report["steps"].append({"sig": tsig, "label": label, "skipped": str(e)[:120]})
                visited.add(tsig)
                steps += 1
                continue

            new_errs = console_errors[before_errs:]
            if new_errs:
                report["errors"].append({"after_click": label, "errors": new_errs})
            try:
                page.screenshot(path=str(SHOTS_DIR / f"step_{steps:03d}.png"))
            except Exception:
                pass
            report["steps"].append({"sig": tsig, "label": label, "new_errors": len(new_errs)})
            visited.add(tsig)
            report["new_elements"] += 1
            steps += 1

            # Return home if we navigated away, so we keep exploring the app.
            if not page.url.rstrip("/").startswith(url.rstrip("/")):
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(500)

        browser.close()

    report["finished"] = time.time()
    report["total_console_errors"] = sum(len(e["errors"]) for e in report["errors"])
    _save_state(visited)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="RagArt exploratory UI crawler")
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--reset", action="store_true", help="forget previously-visited elements")
    args = ap.parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()

    report = explore(args.url, headed=args.headed, max_steps=args.max_steps)
    print(f"Explored {report['new_elements']} new element(s) this run.")
    print(f"Console/JS errors found: {report['total_console_errors']}")
    print(f"Report: {REPORT_PATH}  |  state: {STATE_PATH}")
    if report["total_console_errors"]:
        raise SystemExit(1)  # fail CI if the crawl surfaced JS errors


if __name__ == "__main__":
    main()
