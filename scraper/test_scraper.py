import os
import time
import csv
import json
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_test_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    prefs = {"profile.default_content_setting_values": {"images": 2}}
    options.add_experimental_option("prefs", prefs)
    driver_path = ChromeDriverManager().install()
    if driver_path and ("THIRD_PARTY_NOTICES" in driver_path or "LICENSE" in driver_path):
        parent_dir = os.path.dirname(driver_path)
        actual_path = os.path.join(parent_dir, "chromedriver")
        if os.path.exists(actual_path):
            driver_path = actual_path
    service = Service(driver_path)
    return webdriver.Chrome(service=service, options=options)

def test_extract_ad(driver, ad_url):
    driver.get(ad_url)
    time.sleep(2)
    
    ad_data = {
        "brand_model": None,
        "year": None,
        "mileage": None,
        "price": None,
        "insurance": None,
        "gearbox_type": None,
        "gearbox_health": None,
        "motor_status": None,
        "body_status": None,
        "chassis_status": None,
        "url": ad_url,
    }
    
    try:
        general_rows = driver.find_elements(By.CSS_SELECTOR, '.kt-unexpandable-row')
        for row in general_rows:
            try:
                title = row.find_element(By.CSS_SELECTOR, '.kt-unexpandable-row__title').text.strip()
                value = row.find_element(By.CSS_SELECTOR, '.kt-unexpandable-row__value-box').text.strip()
                if "برند و مدل" in title: ad_data["brand_model"] = value
                elif "قیمت" in title: ad_data["price"] = value
                elif "بیمه" in title: ad_data["insurance"] = value
                elif "گیربکس" in title: ad_data["gearbox_type"] = value
            except Exception: pass

        score_titles = driver.find_elements(By.CSS_SELECTOR, '.kt-score-row__title')
        score_values = driver.find_elements(By.CSS_SELECTOR, '.kt-score-row__score')
        if len(score_titles) == len(score_values):
            for i in range(len(score_titles)):
                title = score_titles[i].text.strip()
                value = score_values[i].text.strip()
                if "موتور" in title: ad_data["motor_status"] = value
                elif "شاسی" in title: ad_data["chassis_status"] = value
                elif "بدنه" in title: ad_data["body_status"] = value
                elif "گیربکس" in title: ad_data["gearbox_health"] = value

        titles = driver.find_elements(By.CSS_SELECTOR, '.kt-group-row-item__title')
        values = driver.find_elements(By.CSS_SELECTOR, '.kt-group-row-item__value')
        for t, v in zip(titles, values):
            try:
                title_text = t.text.strip()
                value_text = v.text.strip()
                if "مدل" in title_text: ad_data["year"] = value_text
                elif "کارکرد" in title_text: ad_data["mileage"] = value_text
            except Exception as e:
                print(f"[!] Error reading group item: {e}")

    except Exception as e:
        print(f"[!] Error reading DOM: {e}")
        
    return ad_data

def save_test_to_csv(dataset):
    if not dataset:
        return
        
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'test_scraper_data.csv')
    
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = dataset[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"\n[✔] Test data saved to {filepath}")

def run_test():
    print("🚀 Starting Test Scraper...")
    driver = setup_test_driver()
    
    try:
        test_url = "https://divar.ir/s/iran/car/pride/131"
        driver.get(test_url)
        time.sleep(3)
        
        links = []
        elements = driver.find_elements(By.CSS_SELECTOR, 'article a')
        for el in elements:
            href = el.get_attribute("href")
            if href and "/v/" in href:
                links.append(href)
            if len(links) == 5: 
                break
                
        print(f"[+] Found {len(links)} links. Starting extraction...\n")
        
        test_dataset = []
        for idx, link in enumerate(links):
            print(f"--- Ad {idx + 1} ---")
            data = test_extract_ad(driver, link)
            test_dataset.append(data)
            print(json.dumps(data, ensure_ascii=False, indent=4))
            print("\n")
            
        save_test_to_csv(test_dataset)
            
    finally:
        driver.quit()
        print("🏁 Test finished.")

if __name__ == "__main__":
    run_test()