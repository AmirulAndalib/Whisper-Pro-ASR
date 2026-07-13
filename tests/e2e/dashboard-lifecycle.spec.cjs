const { test, expect } = require("@playwright/test");

async function setScenario(request, scenario) {
  await request.post("/__reset");
  const res = await request.post("/__lifecycle/scenario", { data: { name: scenario } });
  expect(res.ok()).toBeTruthy();
}

async function advanceLifecycle(request, delta = 1) {
  const res = await request.post("/__lifecycle/advance", { data: { delta } });
  expect(res.ok()).toBeTruthy();
}

function taskCard(page, taskId) {
  return page.locator(`#task-list .task-card[data-task-id="${taskId}"]`);
}

function historyCard(page, taskId) {
  return page.locator(`#history-list .history-card`).filter({ has: page.locator(`text=${taskId}`) });
}

test.describe("Dashboard endpoint lifecycle", () => {
  test("detect-language lifecycle renders correctly in active and history", async ({ page, request }) => {
    await setScenario(request, "lifecycle-detectlang");

    await page.goto("/dashboard");

    // Tick 0: queued + paused-for-priority hint
    const detectCard = taskCard(page, "dl-life-1");
    await expect(detectCard).toHaveCount(1);
    await expect(detectCard).toContainText("detect-life.mkv");
    await expect(detectCard).toContainText("queue");
    await expect(detectCard).toContainText("Paused for priority detect-language tasks");

    // Tick 1: queued + waiting-for-hardware hint
    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(detectCard).toContainText("Waiting for available hardware unit");

    // Tick 2: active language detection
    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(detectCard).toContainText("active");
    await expect(detectCard).toContainText("Language Detection");

    // Tick 3: post-processing
    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(detectCard).toContainText("post-processing");
    await expect(detectCard).toContainText("Post-Processing");

    // Tick 4: removed from active; appears in history
    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(detectCard).toHaveCount(0);

    await page.click("#tab-history");
    await expect(page.locator("#history-list")).toContainText("detect-life.mkv");
    await expect(page.locator("#history-list")).toContainText("EN");
  });

  test("asr lifecycle renders correctly in active and history", async ({ page, request }) => {
    await setScenario(request, "lifecycle-asr");

    await page.goto("/dashboard");

    const asrCard = taskCard(page, "asr-life-1");
    await expect(asrCard).toHaveCount(1);
    await expect(asrCard).toContainText("asr-life.mkv");
    await expect(asrCard).toContainText("queue");
    await expect(asrCard).toContainText("Waiting for available hardware unit");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(asrCard).toContainText("active");
    await expect(asrCard).toContainText("Inference");
    await expect(asrCard).toContainText("LIVE SRT STREAM");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(asrCard).toContainText("post-processing");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(asrCard).toHaveCount(0);

    await page.click("#tab-history");
    await expect(page.locator("#history-list")).toContainText("asr-life.mkv");
    await expect(page.locator("#history-list")).toContainText("asr lifecycle text");
  });

  test("v1 audio lifecycle renders correctly in active and history", async ({ page, request }) => {
    await setScenario(request, "lifecycle-v1");

    await page.goto("/dashboard");

    const v1Card = taskCard(page, "v1-life-1");
    await expect(v1Card).toHaveCount(1);
    await expect(v1Card).toContainText("v1-life.mkv");
    await expect(v1Card).toContainText("queue");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(v1Card).toContainText("active");
    await expect(v1Card).toContainText("Inference");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(v1Card).toContainText("post-processing");

    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });
    await expect(v1Card).toHaveCount(0);

    await page.click("#tab-history");
    await expect(page.locator("#history-list")).toContainText("v1-life.mkv");
    await expect(page.locator("#history-list")).toContainText("v1 lifecycle text");
  });

  test("mixed lifecycle preserves deterministic active-queued ordering", async ({ page, request }) => {
    await setScenario(request, "lifecycle-mixed");

    await page.goto("/dashboard");

    const cards = page.locator("#task-list .task-card");
    await expect(cards).toHaveCount(4);

    await expect(cards.nth(0)).toContainText("mix-active-a.mkv");
    await expect(cards.nth(1)).toContainText("mix-active-b.mkv");
    await expect(cards.nth(2)).toContainText("mix-priority-queued.mkv");
    await expect(cards.nth(3)).toContainText("mix-standard-queued.mkv");

    await expect(cards.nth(2)).toContainText("Paused for priority detect-language tasks");
    await expect(cards.nth(3)).toContainText("Waiting for available hardware unit");
  });
});
