TEST_URL = "http://127.0.0.1:8502/test"
TEST_PASSWORD = "test123"


def test_app_loads(page):
    page.goto(TEST_URL)
    page.wait_for_selector("text=Worklog", timeout=15000)
    assert page.locator("text=Worklog").count() > 0


def test_login_screen_shows(page):
    page.goto(TEST_URL)
    page.wait_for_selector("text=Worklog", timeout=15000)
    assert page.locator("text=Login").count() > 0 or page.locator("text=Password").count() > 0


def test_login_submit(page):
    page.goto(TEST_URL)
    page.wait_for_selector("text=Worklog", timeout=15000)

    password_box = page.locator('input[type="text"], input[type="password"]').last
    password_box.fill(TEST_PASSWORD)

    page.get_by_role("button", name="Login").click()
    page.wait_for_timeout(2000)

    assert page.locator("text=Login failed").count() == 0
    assert (
        page.locator("text=Add New Job").count() > 0
        or page.locator("text=Daily Report").count() > 0
        or page.locator("text=Weekly Report").count() > 0
        or page.locator("text=Monthly Report").count() > 0
    )
