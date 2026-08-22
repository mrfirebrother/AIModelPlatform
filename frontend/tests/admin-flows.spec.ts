import { test, expect } from "@playwright/test";

test.describe("AI 模型服务平台管理后台", () => {
  test("总览页面显示 KPI 和资源状态", async ({ page }) => {
    await page.goto("/dashboard");

    // KPI cards visible
    await expect(page.locator(".kpi")).toHaveCount(4);
    await expect(page.locator(".kpi-label").first()).toBeVisible();

    // GPU resources card
    await expect(page.locator(".card-title").filter({ hasText: "GPU 资源" })).toBeVisible();

    // Training queue card
    await expect(page.locator(".card-title").filter({ hasText: "训练队列" })).toBeVisible();
  });

  test("侧边栏导航切换页面", async ({ page }) => {
    await page.goto("/");

    // Click models navigation
    await page.click("text=模型谱系");
    await expect(page.locator(".card-title").filter({ hasText: "根模型谱系" })).toBeVisible();

    // Click bindings
    await page.click("text=模型关联");
    await expect(page.locator(".card-title").filter({ hasText: "模型关联列表" })).toBeVisible();

    // Click datasets
    await page.click("text=数据集");
    await expect(page.locator(".card-title").filter({ hasText: "数据集管理" })).toBeVisible();

    // Click training
    await page.click("text=训练任务");
    await expect(page.locator(".card-title").filter({ hasText: "训练任务队列" })).toBeVisible();

    // Click evaluations
    await page.click("text=模型评估");
    await expect(page.locator(".card-title").filter({ hasText: "评估收件箱" })).toBeVisible();
  });

  test("模型谱系树展示节点", async ({ page }) => {
    await page.goto("/models");

    // Root model nodes should be visible
    await expect(page.locator(".tree-node.root")).toHaveCount(2);
    await expect(page.locator(".node-name").filter({ hasText: "YOLO 基础权重" })).toBeVisible();
  });

  test("模型详情页展示信息", async ({ page }) => {
    await page.goto("/models/m-bridge-a-concrete");

    await expect(page.locator(".card-title").filter({ hasText: "模型节点详情" })).toBeVisible();
    await expect(page.locator(".detail-row").filter({ hasText: "模型名称" })).toBeVisible();
    await expect(page.locator(".detail-row").filter({ hasText: "A 桥 / 混凝土" })).toBeVisible();
  });

  test("模型关联列表展示数据", async ({ page }) => {
    await page.goto("/bindings");

    await expect(page.locator("th").filter({ hasText: "关联 ID" })).toBeVisible();
    await expect(page.locator("td").filter({ hasText: "mb-bridge-a-concrete-001" })).toBeVisible();
  });

  test("数据集页面展示数据", async ({ page }) => {
    await page.goto("/datasets");

    await expect(page.locator("th").filter({ hasText: "数据集名称" })).toBeVisible();
    await expect(page.locator("td").filter({ hasText: "A 桥混凝土数据集" })).toBeVisible();
  });

  test("训练任务页面展示队列", async ({ page }) => {
    await page.goto("/training");

    await expect(page.locator(".card-title").filter({ hasText: "训练任务队列" })).toBeVisible();
    await expect(page.locator("th").filter({ hasText: "任务 ID" })).toBeVisible();
  });

  test("新建训练任务表单", async ({ page }) => {
    await page.goto("/training/new");

    await expect(page.locator(".card-title").filter({ hasText: "新建训练任务" })).toBeVisible();
    await expect(page.locator("label").filter({ hasText: "父模型节点" })).toBeVisible();
    await expect(page.locator("label").filter({ hasText: "数据集快照" })).toBeVisible();
    await expect(page.locator("label").filter({ hasText: "训练轮数" })).toBeVisible();
  });

  test("评估页面展示收件箱", async ({ page }) => {
    await page.goto("/evaluations");

    await expect(page.locator(".card-title").filter({ hasText: "评估收件箱" })).toBeVisible();
    await expect(page.locator("td").filter({ hasText: "工业园区 / 火灾" })).toBeVisible();
  });

  test("发布记录页面", async ({ page }) => {
    await page.goto("/releases");

    await expect(page.locator(".card-title").filter({ hasText: "发布与回滚记录" })).toBeVisible();
    await expect(page.locator("th").filter({ hasText: "模型节点" })).toBeVisible();
  });

  test("资源控制页面", async ({ page }) => {
    await page.goto("/resources");

    await expect(page.locator(".card-title").filter({ hasText: "GPU 资源总览" })).toBeVisible();
    await expect(page.locator(".card-title").filter({ hasText: "资源分配策略" })).toBeVisible();
  });
});
