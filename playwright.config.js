// Playwright config for RagArt web E2E tests.
//   npm install && npx playwright install chromium
//   npm run test:e2e          (auto-starts `ragart` if not already running)
//
// The webServer block boots RagArt on :5000 and waits for /health. First boot
// downloads the embedding model (~470 MB), so the timeout is generous. If you
// already have a server running, it's reused.
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.RAGART_URL || 'http://localhost:5000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'ragart --no-browser --host 127.0.0.1 --port 5000',
    url: 'http://localhost:5000/health',
    reuseExistingServer: true,
    timeout: 300_000,  // first boot downloads the embedding model
  },
});
