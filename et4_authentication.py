# -*- coding: utf-8 -*-
"""
Zerodha kiteconnect automated authentication
"""

from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from pyotp import TOTP
import pandas as pd

# Define the correct path
cwd = "C:\\Users\\Saarit\\OneDrive\\Desktop\\Trading\\et4\\trading_strategies_algo"


def autologin():
    token_path = "api_key.txt"
    key_secret = open(token_path, 'r').read().split()
    kite = KiteConnect(api_key=key_secret[0])

    # Setup Chrome WebDriver
    service = Service(ChromeDriverManager().install())
    options = Options()
    # Uncomment this if you want to run headless
    # options.add_argument('--headless')

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(kite.login_url())

    # Use WebDriverWait to ensure elements are present
    wait = WebDriverWait(driver, 10)

    def retry_element(by, value, retries=3):
        """
        A helper function to retry locating an element if it goes stale.
        :param by: Locator strategy (By.XPATH, etc.)
        :param value: Locator value
        :param retries: Number of retry attempts
        :return: The located web element
        """
        attempt = 0
        while attempt < retries:
            try:
                element = wait.until(EC.presence_of_element_located((by, value)))
                return element
            except StaleElementReferenceException:
                attempt += 1
                time.sleep(2)  # Short delay before retry
                print(f"Retrying element location: {attempt}/{retries}")
        raise StaleElementReferenceException(f"Could not locate element after {retries} retries")

    # Login Process
    username = retry_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
    password = retry_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input')
    username.send_keys(key_secret[2])
    password.send_keys(key_secret[3])

    # Click login button
    retry_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()

    # Enter TOTP PIN
    pin = retry_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[1]/input')
    totp = TOTP(key_secret[4])
    token = totp.now()
    pin.send_keys(token)

    # Submit the form
    retry_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[2]/button').click()

    time.sleep(5)  # Wait for the page to redirect and token to appear

    # Extract request token
    try:
        request_token = driver.current_url.split('request_token=')[1][:32]
    except StaleElementReferenceException:
        print("Encountered stale element. Retrying...")
        time.sleep(2)  # Delay before retrying
        request_token = driver.current_url.split('request_token=')[1][:32]

    # Save request token to file
    with open('request_token.txt', 'w') as the_file:
        the_file.write(request_token)
    driver.quit()

autologin()

# Generate and store access token - valid till 6 AM the next day
request_token = open("request_token.txt", 'r').read()
key_secret = open("api_key.txt", 'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
data = kite.generate_session(request_token, api_secret=key_secret[1])

with open('access_token.txt', 'w') as file:
    file.write(data["access_token"])


# Get dump of all NSE instruments
nse_instrument_dump = kite.instruments("NSE")
nse_instrument_df = pd.DataFrame(nse_instrument_dump)

# Get dump of all BSE instruments
bse_instrument_dump = kite.instruments("BSE")
bse_instrument_df = pd.DataFrame(bse_instrument_dump)

# Merge the NSE and BSE data
merged_instrument_df = pd.concat([nse_instrument_df, bse_instrument_df])

# Save the merged data to a single CSV
merged_instrument_df.to_csv("Merged_NSE_BSE_Instruments.csv", index=False)

