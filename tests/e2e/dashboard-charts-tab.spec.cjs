const { test, expect } = require("@playwright/test");

test.describe("Dashboard charts tab", () => {
  test("shows chart tab telemetry panels and refreshes chart stats", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-charts");

    await expect(page.locator("#charts-section")).toBeVisible();
    await expect(page.locator("#cpuChart")).toBeVisible();
    await expect(page.locator("#memChart")).toBeVisible();
    await expect(page.locator("#hwChart")).toBeVisible();

    await expect(page.locator("#cpu-sys-current")).not.toHaveText("0%");
    await expect(page.locator("#cpu-app-current")).not.toHaveText("0%");
    await expect(page.locator("#mem-app-current")).not.toHaveText("0 GB");
  });

  test("allows changing chart window without breaking telemetry view", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-charts");

    await page.selectOption("#chart-window-select", "3");
    await expect(page.locator("#chart-window-select")).toHaveValue("3");

    await page.selectOption("#chart-window-select", "10");
    await expect(page.locator("#chart-window-select")).toHaveValue("10");
    await expect(page.locator("#cpu-sys-highest")).toContainText("%");
  });
});
