const { test, expect } = require("@playwright/test");
const fs = require("fs");

test.describe("Analytics dashboard", () => {
  test("renders cumulative metrics and detect-lang fallback data", async ({ page }) => {
    await page.goto("/analytics");

    await expect(page.getByRole("heading", { name: "Whisper Pro Analytics" })).toBeVisible();
    await expect(page.locator("#val-total-tasks")).toHaveText("15");
    await expect(page.locator("#val-today-tasks")).toHaveText("4");

    // detectlang card should use fallback from cumulative.isolations when detectlang is missing.
    await expect(page.locator("#val-detectlang-count")).toHaveText("3 tasks");

    // Daily table is rendered newest first.
    const firstDateCell = page.locator("#table-body tr").first().locator("td").first();
    await expect(firstDateCell).toHaveText("2026-07-14");
  });

  test("renders charts and supports back navigation", async ({ page }) => {
    await page.goto("/analytics");

    await expect(page.locator("#tasksChart")).toBeVisible();
    await expect(page.locator("#durationChart")).toBeVisible();
    await expect(page.locator("#last-update")).toContainText("Updated:");

    await page.click("#btn-back");
    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.locator("#tab-active")).toBeVisible();
  });

  test("exports analytics data as downloadable json", async ({ page }) => {
    await page.goto("/analytics");
    await expect(page.locator("#val-total-tasks")).toHaveText("15");

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click("#btn-export"),
    ]);

    expect(download.suggestedFilename()).toBe("whisper_pro_analytics.json");

    const downloadPath = await download.path();
    expect(downloadPath).toBeTruthy();

    const raw = fs.readFileSync(downloadPath, "utf8");
    const exported = JSON.parse(raw);
    expect(exported).toHaveProperty("cumulative");
    expect(exported).toHaveProperty("daily");
    expect(exported.cumulative.count_all_time).toBe(15);
  });
});
