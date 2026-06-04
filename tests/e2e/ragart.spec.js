// Deterministic web E2E for the RagArt UI. These exercise UI behaviour that
// doesn't depend on the (slow, model-loading) RAG backend being "ready", so
// they're fast and CI-friendly. The full ask→stream flow is covered by the
// Python integration suite (tests/integration/) and the exploratory crawler
// (scripts/agent_explore.py).
const { test, expect } = require('@playwright/test');

test.describe('RagArt UI smoke', () => {
  test('page loads with no console errors', async ({ page }) => {
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto('/');
    await expect(page).toHaveTitle(/RagArt/i);
    // Ignore benign favicon/network noise; fail on real JS errors.
    const real = errors.filter((e) => !/favicon|net::ERR/i.test(e));
    expect(real, real.join('\n')).toHaveLength(0);
  });

  test('consent notice appears and dismisses', async ({ page }) => {
    await page.goto('/');
    const consent = page.locator('#consentBar');
    await expect(consent).toBeVisible();
    await consent.getByRole('button', { name: 'Tamam' }).click();
    await expect(consent).toBeHidden();
  });

  test('feedback widget opens, validates and submits', async ({ page }) => {
    await page.goto('/');
    await page.locator('#feedbackFab').click();
    const modal = page.locator('#fbModal');
    await expect(modal).toBeVisible();
    // Pick 4 stars, type a message, send.
    await modal.locator('span[data-i="4"]').click();
    await modal.locator('textarea').fill('Playwright test geri bildirimi');
    await modal.getByRole('button', { name: 'Gönder' }).click();
    await expect(modal).toBeHidden();
  });

  test('provider selection populates the model dropdown', async ({ page }) => {
    await page.goto('/');
    // Open Settings (the gear / settings nav). Fall back to the route hash.
    await page.evaluate(() => { location.hash = '#settings'; });
    const provider = page.locator('#setProvider');
    await provider.waitFor({ state: 'attached' });
    await provider.selectOption('anthropic');
    const modelSelect = page.locator('#setModelSelect');
    await expect(modelSelect).toBeVisible();
    // Anthropic's curated models must be offered.
    const options = await modelSelect.locator('option').allInnerTexts();
    expect(options.join(' ')).toMatch(/claude/i);
    // The "type your own" escape hatch exists.
    expect(options.join(' ')).toMatch(/Özel model/i);
  });

  test('question input is present', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#qInput')).toBeVisible();
    await expect(page.locator('#askBtn')).toBeVisible();
  });
});
