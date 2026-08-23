from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:8080/datasets")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)
    page.screenshot(path="C:\\Users\\ThinkPad\\AppData\\Local\\Temp\\opencode\\datasets-page.png", full_page=True)
    print("Screenshot saved")
    browser.close()
