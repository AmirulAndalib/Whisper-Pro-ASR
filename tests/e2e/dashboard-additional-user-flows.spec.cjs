const { test, expect } = require("@playwright/test");
const fs = require("fs");

async function setScenario(request, scenario) {
  await request.post("/__reset");
  const res = await request.post("/__lifecycle/scenario", { data: { name: scenario } });
  expect(res.ok()).toBeTruthy();
}

async function advanceLifecycle(request, delta = 1) {
  const res = await request.post("/__lifecycle/advance", { data: { delta } });
  expect(res.ok()).toBeTruthy();
}

test.describe("Dashboard additional user flows", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("/__reset");
  });

  test("shows idle active state and history no-match empty state", async ({ page, request }) => {
    await setScenario(request, "lifecycle-asr");

    await page.goto("/dashboard");

    // Move lifecycle to completion so active list is empty.
    await advanceLifecycle(request, 3);
    await page.evaluate(async () => {
      await window.updateStats();
    });

    await expect(page.locator("#task-list")).toContainText("Service is idle");

    await page.click("#tab-history");
    await expect(page.locator("#history-list")).toContainText("asr-life.mkv");

    await page.click("#hist-filter-detectlang");
    await expect(page.locator("#history-list")).toContainText("No history matches filter");
  });

  test("updates settings range labels on input", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "72");
    await expect(page.locator("#retention-label")).toHaveText("72h");

    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "21");
    await expect(page.locator("#log-retention-label")).toHaveText("21d");
  });

  test("downloads SRT from history transcription details", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-history");

    const asrCard = page.locator("#history-list .history-card").filter({ hasText: "history-asr.mkv" });
    await expect(asrCard).toHaveCount(1);

    await asrCard.locator("summary", { hasText: "View Transcription Result" }).click();

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      asrCard.getByRole("button", { name: "SRT" }).click(),
    ]);

    expect(download.suggestedFilename()).toBe("history-asr.srt");
    const filePath = await download.path();
    expect(filePath).toBeTruthy();

    const raw = fs.readFileSync(filePath, "utf8");
    expect(raw).toContain("asr text");
  });

  test("navigates from dashboard analytics tab to full analytics page", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-analytics");

    await expect(page.locator("#btn-full-analytics")).toBeVisible();
    await page.click("#btn-full-analytics");

    await expect(page).toHaveURL(/\/analytics$/);
    await expect(page.getByRole("heading", { name: "Whisper Pro Analytics" })).toBeVisible();
  });
});
