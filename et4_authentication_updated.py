# -*- coding: utf-8 -*-
"""
Zerodha KiteConnect automated authentication (resilient‑v4)
---------------------------------------------------------
Fix for “stuck after TOTP”
• **Scans every open window/tab** for a URL that already contains `request_token=` – sometimes Kite opens a new tab instead of re‑using the first one.
• Beefed‑up “allow / authorise” click routine – wider set of selectors, loops for up to 15 s instead of a single 5 s hit.
• Verbose `print()` breadcrumbs so you can see exactly which step it’s on while debugging.
"""

import os
import time
import contextlib
from typing import Iterable, Tuple

from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
)
from webdriver_manager.chrome import ChromeDriverManager
from pyotp import TOTP
import pandas as pd

# ───────────────────────── CONFIG ─────────────────────────
CWD = r"C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"
API_FILE = "api_key.txt"          # api_key api_secret user_id password totp_secret
REQUEST_FILE = "request_token.txt"
ACCESS_FILE = "access_token.txt"
INSTRUMENTS_FILE = "Merged_NSE_BSE_Instruments.csv"
PAGE_TIMEOUT = 30
MAX_RETRIES = 5
REDIRECT_WAIT = 60                 # max seconds to wait for redirect

# ────────────────────── HELPER FUNCTION ───────────────────

def do_action(driver: webdriver.Chrome,
              wait: WebDriverWait,
              locators: Iterable[Tuple[str, str]],
              action: str,
              text: str | None = None) -> None:
    """Try each locator with retries; raise TimeoutException on failure."""
    for by, sel in locators:
        for _ in range(1, MAX_RETRIES + 1):
            try:
                if action == "click":
                    elem = wait.until(EC.element_to_be_clickable((by, sel)))
                    if elem is None:
                        raise StaleElementReferenceException
                    elem.click()
                elif action == "send_keys":
                    elem = wait.until(EC.presence_of_element_located((by, sel)))
                    if elem is None:
                        raise StaleElementReferenceException
                    elem.clear()
                    elem.send_keys(text)
                else:
                    raise ValueError("Unknown action")
                return
            except (StaleElementReferenceException, TimeoutException):
                time.sleep(0.8)
    raise TimeoutException(f"All locators failed for {action}")


# ───────────────────────── MAIN FLOW ──────────────────────

def kite_autologin(headless: bool = False):
    os.chdir(CWD)
    api_key, api_secret, user_id, pwd, totp_secret = open(API_FILE).read().split()
    kite = KiteConnect(api_key=api_key)

    service = Service(ChromeDriverManager().install())
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, PAGE_TIMEOUT)

    with contextlib.suppress(Exception):
        driver.set_window_size(1280, 900)

    try:
        print("→ Opening login page…")
        driver.get(kite.login_url())

        # 1️⃣ User‑ID & Password
        print("→ Entering credentials…")
        do_action(driver, wait, [
            (By.CSS_SELECTOR, "input#userid"),
            (By.CSS_SELECTOR, "input[name='user_id']"),
            (By.XPATH, "//input[@placeholder='User ID']"),
        ], "send_keys", user_id)

        do_action(driver, wait, [
            (By.CSS_SELECTOR, "input#password"),
            (By.CSS_SELECTOR, "input[name='password']"),
            (By.XPATH, "//input[@placeholder='Password']"),
        ], "send_keys", pwd)

        do_action(driver, wait, [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//button[contains(.,'Login')]"),
        ], "click")

        # 2️⃣ TOTP
        print("→ Submitting TOTP…")
        do_action(driver, wait, [
            (By.CSS_SELECTOR, "input#pin"),
            (By.CSS_SELECTOR, "input[name='pin']"),
            (By.XPATH, "//input[@placeholder='PIN']"),
        ], "send_keys", TOTP(totp_secret).now())

        do_action(driver, wait, [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//button[contains(.,'Continue') or contains(.,'Submit')]"),
        ], "click")

        # 2.5️⃣ Allow / Authorise (if shown)
        print("→ Checking for consent screen…")
        consent_locators = [
            (By.CSS_SELECTOR, "button#authorize"),
            (By.CSS_SELECTOR, "button.button-orange"),
            (By.CSS_SELECTOR, "a.button-orange"),
            (By.XPATH, "//button[contains(.,'Allow') or contains(.,'Authorise') or contains(.,'Authorize')]") ,
            (By.XPATH, "//a[contains(.,'Allow') or contains(.,'Authorize')]"),
        ]
        consent_clicked = False
        start_consent = time.time()
        while time.time() - start_consent < 15 and not consent_clicked:
            try:
                do_action(driver, WebDriverWait(driver, 5), consent_locators, "click")
                consent_clicked = True
                print("→ Consent screen handled.")
            except TimeoutException:
                # Maybe consent page never appeared – break after 15 s
                break

        # 3️⃣ Wait / scan for request_token in any tab
        print("→ Waiting for redirect with request_token…")
        req_token = None
        start = time.time()
        while time.time() - start < REDIRECT_WAIT and req_token is None:
            for handle in driver.window_handles:
                driver.switch_to.window(handle)
                if "request_token=" in driver.current_url:
                    req_token = driver.current_url.split("request_token=")[1].split("&")[0]
                    break
            time.sleep(1)

        if req_token is None:
            raise RuntimeError("request_token not found within timeout")

        print("✓ request_token captured →", req_token)
        with open(REQUEST_FILE, "w") as f:
            f.write(req_token)
    finally:
        driver.quit()

    # 4️⃣ Exchange for access_token
    session = kite.generate_session(req_token, api_secret=api_secret)
    with open(ACCESS_FILE, "w") as f:
        f.write(session["access_token"])
    print("✓ access_token saved →", session["access_token"][:10], "… ")

    # 5️⃣ Dump instruments
    kite.set_access_token(session["access_token"])
    pd.concat([
        pd.DataFrame(kite.instruments("NSE")),
        pd.DataFrame(kite.instruments("BSE")),
    ]).to_csv(INSTRUMENTS_FILE, index=False)
    print("✓ instruments CSV written →", INSTRUMENTS_FILE)


if __name__ == "__main__":
    kite_autologin(headless=False)
