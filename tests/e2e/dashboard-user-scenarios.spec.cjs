const { test, expect } = require("@playwright/test");

test.describe("Dashboard user scenarios", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("/__reset");
  });

  test("does not expose a manual refresh interval option", async ({ page }) => {
    await page.goto("/dashboard");
    await expect(page.locator("#refresh-interval option[value='manual']")).toHaveCount(0);
  });

  test("shows deterministic task ordering and paused-for-priority messaging", async ({ page }) => {
    await page.goto("/dashboard");

    const taskCards = page.locator("#task-list .task-card");
    await expect(taskCards).toHaveCount(3);

    await expect(taskCards.nth(0)).toContainText("movie-asr.mkv");
    await expect(taskCards.nth(1)).toContainText("movie-detect.mkv");
    await expect(taskCards.nth(2)).toContainText("movie-v1.mkv");

    await expect(taskCards.nth(1)).toContainText("Paused for priority detect-language tasks");
    await expect(taskCards.nth(2)).toContainText("Waiting for available hardware unit");
  });

  test("pauses and resumes live refresh polling", async ({ page, request }) => {
    await page.goto("/dashboard");

    const beforePauseResp = await request.get("/__events");
    const beforePauseEvents = await beforePauseResp.json();
    const refreshIntervalMs = Number(await page.locator("#refresh-interval").inputValue());

    // Pause live refresh and verify polling stops (allow one in‑flight request)
    await page.click("#toggle-refresh");
    const pausedBaselineResp = await request.get("/__events");
    const pausedBaselineEvents = await pausedBaselineResp.json();
    expect(pausedBaselineEvents.statusCalls).toBeLessThanOrEqual(beforePauseEvents.statusCalls + 1);

    await page.waitForTimeout(refreshIntervalMs * 2 + 250);

    // Capture paused state after confirming polling stopped
    const pausedResp = await request.get("/__events");
    const pausedEvents = await pausedResp.json();
    expect(pausedEvents.statusCalls).toBe(pausedBaselineEvents.statusCalls);

    // Resume live refresh and verify polling restarts
    await page.click("#toggle-refresh");
    await expect.poll(async () => {
      const resp = await request.get("/__events");
      const data = await resp.json();
      return data.statusCalls;
    }).toBeGreaterThan(pausedEvents.statusCalls);

    // Capture resumed state and assert
    const resumedResp = await request.get("/__events");
    const resumedEvents = await resumedResp.json();
    expect(resumedEvents.statusCalls).toBeGreaterThan(pausedEvents.statusCalls);
  });

  test("saves settings and executes maintenance actions", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "48");
    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "14");

    page.on("dialog", (dialog) => dialog.accept());

    await page.getByRole("button", { name: "Save Configuration" }).click();
    await page.getByRole("button", { name: "Clear History" }).click();
    await page.getByRole("button", { name: "Clear Telemetry" }).click();

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();

    expect(events.settingsSaves.length).toBeGreaterThan(0);
    expect(events.settingsSaves[events.settingsSaves.length - 1]).toMatchObject({
      telemetry_retention_hours: 48,
      log_retention_days: 14,
    });
    expect(events.historyClears).toBeGreaterThan(0);
    expect(events.telemetryClears).toBeGreaterThan(0);
  });

  test("downloads logs from dashboard header", async ({ page, request }) => {
    await page.goto("/dashboard");

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click("a[href='/logs/download']"),
    ]);

    expect(download.suggestedFilename()).toBe("whisper_pro.log");

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();
    expect(events.logDownloads).toBeGreaterThan(0);
  });
});
