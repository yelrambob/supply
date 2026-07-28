"""Load the Streamlit app in a real browser and click through the sleep screen if present.

A plain HTTP ping doesn't keep a Streamlit Community Cloud app awake because the
"gone to sleep" screen only wakes up in response to a real page load (with the
"Yes, get this app back up!" button clicked), not a bare curl request.
"""
import sys

from playwright.sync_api import sync_playwright

APP_URL = "https://ordered.streamlit.app/"
WAKE_BUTTON_TEXT = "get this app back up"
READY_TEXT = "Supply Ordering & Inventory Tracker"


def wake_app() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, timeout=60_000, wait_until="networkidle")

        wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT, exact=False)
        if wake_button.count() > 0:
            print("App is asleep, clicking the wake button...")
            wake_button.first.click()
        else:
            print("No wake button found; app may already be awake.")

        try:
            page.get_by_text(READY_TEXT).wait_for(timeout=120_000)
            print("App is awake and loaded.")
        except Exception as exc:
            print(f"Timed out waiting for app to load: {exc}")
            browser.close()
            sys.exit(1)

        browser.close()


if __name__ == "__main__":
    wake_app()
