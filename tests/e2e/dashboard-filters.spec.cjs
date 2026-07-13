const { test, expect } = require("@playwright/test");

async function visibleTaskCards(page) {
  return page.locator("#task-list .task-card").count();
}

async function visibleHistoryCards(page) {
  return page.locator("#history-list .history-card").count();
}

test.describe("Dashboard endpoint filters", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("/__reset");
  });

  test("renders endpoint filter buttons with expected labels", async ({ page }) => {
    await page.goto("/dashboard");

    await expect(page.locator("#filter-asr")).toHaveText("/asr");
    await expect(page.locator("#filter-detectlang")).toHaveText("/detect-language");
    await expect(page.locator("#filter-v1")).toHaveText("/v1/audio/...");

    await page.click("#tab-history");
    await expect(page.locator("#hist-filter-asr")).toHaveText("/asr");
    await expect(page.locator("#hist-filter-detectlang")).toHaveText("/detect-language");
    await expect(page.locator("#hist-filter-v1")).toHaveText("/v1/audio/...");
  });

  test("filters active tasks by endpoint category", async ({ page }) => {
    await page.goto("/dashboard");

    await expect(page.locator("#task-list")).toContainText("movie-asr.mkv");
    await expect(page.locator("#task-list")).toContainText("movie-detect.mkv");
    await expect(page.locator("#task-list")).toContainText("movie-v1.mkv");
    await expect(await visibleTaskCards(page)).toBe(3);

    await page.click("#filter-asr");
    await expect.poll(() => visibleTaskCards(page)).toBe(1);
    await expect(page.locator("#task-list")).toContainText("movie-asr.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-detect.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-v1.mkv");

    await page.click("#filter-detectlang");
    await expect.poll(() => visibleTaskCards(page)).toBe(1);
    await expect(page.locator("#task-list")).toContainText("movie-detect.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-asr.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-v1.mkv");

    await page.click("#filter-v1");
    await expect.poll(() => visibleTaskCards(page)).toBe(1);
    await expect(page.locator("#task-list")).toContainText("movie-v1.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-asr.mkv");
    await expect(page.locator("#task-list")).not.toContainText("movie-detect.mkv");

    await page.click("#filter-all");
    await expect.poll(() => visibleTaskCards(page)).toBe(3);
  });

  test("filters history entries by endpoint category", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-history");

    await expect(page.locator("#history-list")).toContainText("history-asr.mkv");
    await expect(page.locator("#history-list")).toContainText("history-detect.mkv");
    await expect(page.locator("#history-list")).toContainText("history-v1.mkv");
    await expect.poll(() => visibleHistoryCards(page)).toBe(3);

    await page.click("#hist-filter-asr");
    await expect.poll(() => visibleHistoryCards(page)).toBe(1);
    await expect(page.locator("#history-list")).toContainText("history-asr.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-detect.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-v1.mkv");

    await page.click("#hist-filter-detectlang");
    await expect.poll(() => visibleHistoryCards(page)).toBe(1);
    await expect(page.locator("#history-list")).toContainText("history-detect.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-asr.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-v1.mkv");

    await page.click("#hist-filter-v1");
    await expect.poll(() => visibleHistoryCards(page)).toBe(1);
    await expect(page.locator("#history-list")).toContainText("history-v1.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-asr.mkv");
    await expect(page.locator("#history-list")).not.toContainText("history-detect.mkv");

    await page.click("#hist-filter-all");
    await expect.poll(() => visibleHistoryCards(page)).toBe(3);
  });
});