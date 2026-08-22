import { test, expect } from "@playwright/test";

const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

test.describe("Closed-loop E2E: admin UI flow", () => {
  test("dashboard loads and shows KPI cards", async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`);

    await expect(page.locator(".kpi")).toHaveCount(4);
    await expect(page.locator(".kpi-label").first()).toBeVisible();
    await expect(
      page.locator(".card-title").filter({ hasText: "GPU 资源" })
    ).toBeVisible();
  });

  test("navigate to models tree and view root node", async ({ page }) => {
    await page.goto(`${BASE_URL}/models`);

    await expect(
      page.locator(".card-title").filter({ hasText: "根模型谱系" })
    ).toBeVisible();
    await expect(page.locator(".tree-node.root")).toHaveCount(2);
    await expect(
      page.locator(".node-name").filter({ hasText: "YOLO 基础权重" })
    ).toBeVisible();
  });

  test("model detail page shows node info", async ({ page }) => {
    await page.goto(`${BASE_URL}/models/m-bridge-a-concrete`);

    await expect(
      page.locator(".card-title").filter({ hasText: "模型节点详情" })
    ).toBeVisible();
    await expect(
      page.locator(".detail-row").filter({ hasText: "模型名称" })
    ).toBeVisible();
  });

  test("bindings list displays entries", async ({ page }) => {
    await page.goto(`${BASE_URL}/bindings`);

    await expect(page.locator("th").filter({ hasText: "关联 ID" })).toBeVisible();
    await expect(
      page.locator("td").filter({ hasText: "mb-bridge-a-concrete-001" })
    ).toBeVisible();
  });

  test("datasets page shows data", async ({ page }) => {
    await page.goto(`${BASE_URL}/datasets`);

    await expect(
      page.locator("th").filter({ hasText: "数据集名称" })
    ).toBeVisible();
    await expect(
      page.locator("td").filter({ hasText: "A 桥混凝土数据集" })
    ).toBeVisible();
  });

  test("training tasks page shows queue", async ({ page }) => {
    await page.goto(`${BASE_URL}/training`);

    await expect(
      page.locator(".card-title").filter({ hasText: "训练任务队列" })
    ).toBeVisible();
    await expect(page.locator("th").filter({ hasText: "任务 ID" })).toBeVisible();
  });

  test("new training task form has required fields", async ({ page }) => {
    await page.goto(`${BASE_URL}/training/new`);

    await expect(
      page.locator(".card-title").filter({ hasText: "新建训练任务" })
    ).toBeVisible();
    await expect(
      page.locator("label").filter({ hasText: "父模型节点" })
    ).toBeVisible();
    await expect(
      page.locator("label").filter({ hasText: "数据集快照" })
    ).toBeVisible();
    await expect(
      page.locator("label").filter({ hasText: "训练轮数" })
    ).toBeVisible();
  });

  test("evaluations page shows inbox", async ({ page }) => {
    await page.goto(`${BASE_URL}/evaluations`);

    await expect(
      page.locator(".card-title").filter({ hasText: "评估收件箱" })
    ).toBeVisible();
    await expect(
      page.locator("td").filter({ hasText: "工业园区 / 火灾" })
    ).toBeVisible();
  });

  test("releases page shows history", async ({ page }) => {
    await page.goto(`${BASE_URL}/releases`);

    await expect(
      page.locator(".card-title").filter({ hasText: "发布与回滚记录" })
    ).toBeVisible();
    await expect(
      page.locator("th").filter({ hasText: "模型节点" })
    ).toBeVisible();
  });

  test("resources page shows GPU overview", async ({ page }) => {
    await page.goto(`${BASE_URL}/resources`);

    await expect(
      page.locator(".card-title").filter({ hasText: "GPU 资源总览" })
    ).toBeVisible();
    await expect(
      page.locator(".card-title").filter({ hasText: "资源分配策略" })
    ).toBeVisible();
  });

  test("sidebar navigation covers all main sections", async ({ page }) => {
    await page.goto(`${BASE_URL}/`);

    const sections = [
      { nav: "模型谱系", title: "根模型谱系" },
      { nav: "模型关联", title: "模型关联列表" },
      { nav: "数据集", title: "数据集管理" },
      { nav: "训练任务", title: "训练任务队列" },
      { nav: "模型评估", title: "评估收件箱" },
    ];

    for (const section of sections) {
      await page.click(`text=${section.nav}`);
      await expect(
        page.locator(".card-title").filter({ hasText: section.title })
      ).toBeVisible();
    }
  });
});
