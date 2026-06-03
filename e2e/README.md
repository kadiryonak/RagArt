# RagArt end-to-end testing

Three complementary layers:

| Layer | What | How to run |
|-------|------|-----------|
| **Python integration** | Full HTTP → pipeline → ChromaDB → response, incl. upload/reindex of every file type, streaming, workspace isolation. Fast (fake embedder). | `pytest tests/integration/` |
| **Playwright web E2E** | Deterministic browser tests of the real UI (load, consent, feedback, provider/model dropdown, inputs). | `npm install && npx playwright install chromium && npm run test:e2e` |
| **Agent explorer** | Exploratory crawler that clicks every interactive element it hasn't seen, watches for JS errors, screenshots each step, and **records visited elements so it never repeats them**. | `pip install playwright && playwright install chromium` then `python scripts/agent_explore.py` |

## Playwright (deterministic)

```bash
npm install
npx playwright install chromium
npm run test:e2e          # auto-starts `ragart` (reuses a running one)
npm run test:e2e:ui       # interactive UI mode
npm run test:e2e:report   # open the HTML report
```

First server boot downloads the embedding model (~470 MB) — the config allows
up to 5 minutes for `/health`.

## Agent explorer (exploratory + self-recording)

```bash
python scripts/agent_explore.py --url http://localhost:5000
python scripts/agent_explore.py --headed     # watch it click around
python scripts/agent_explore.py --reset      # forget history, explore fresh
```

- Visited elements are remembered in `e2e/.explore-state.json`, so each run
  only tests **new** interactions.
- A per-run report (actions + any JS/console errors) lands in
  `e2e/explore-report.json`; screenshots in `e2e/explore-shots/`.
- Exits non-zero if the crawl surfaces JS errors (CI-friendly).
