import os
import time
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    
    # options.add_argument('--headless')
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('--disable-extensions')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    prefs = {
        "profile.default_content_setting_values": {
            "images": 2  
        }
    }
    options.add_experimental_option("prefs", prefs)
    
    # Try finding locally cached ChromeDriver first to bypass network and geographic restrictions
    wdm_dir = os.path.expanduser("~/.wdm")
    chromedriver_path = None
    if os.path.exists(wdm_dir):
        for root, dirs, files in os.walk(wdm_dir):
            if "chromedriver" in files:
                full_path = os.path.join(root, "chromedriver")
                if os.access(full_path, os.X_OK):
                    # Check if the found path is actually the chromedriver binary and not a directory with metadata
                    if not ("THIRD_PARTY_NOTICES" in full_path or "LICENSE" in full_path):
                        chromedriver_path = full_path
                        break

    if chromedriver_path:
        driver_path = chromedriver_path
        print(f"[*] Using cached ChromeDriver: {driver_path}")
    else:
        print("[*] Cached ChromeDriver not found. Downloading via ChromeDriverManager...")
        try:
            driver_path = ChromeDriverManager().install()
            if driver_path and ("THIRD_PARTY_NOTICES" in driver_path or "LICENSE" in driver_path):
                parent_dir = os.path.dirname(driver_path)
                actual_path = os.path.join(parent_dir, "chromedriver")
                if os.path.exists(actual_path):
                    driver_path = actual_path
        except Exception as e:
            print(f"\n[!] Failed to download/update ChromeDriver via webdriver-manager: {e}")
            raise e
            
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def get_ad_links(driver, category_url, car_type, max_links=1000):
    print(f"\n[*] [{car_type}] Extracting links from: {category_url}")
    driver.get(category_url)
    time.sleep(3)
    
    links = set()
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while len(links) < max_links:
        elements = driver.find_elements(By.CSS_SELECTOR, 'article a')
        for el in elements:
            try:
                href = el.get_attribute("href")
                if href and "/v/" in href:
                    links.add(href)
            except Exception:
                continue
            
        print(f"[{car_type}] Collected links: {len(links)}/{max_links}", end="\r")
        if len(links) >= max_links:
            break
            
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.5, 3.0))
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print(f"\n[!] [{car_type}] Reached the end of the list.")
            break
        last_height = new_height
        
    print(f"\n[OK] [{car_type}] Successfully extracted {len(links)} links.")
    return list(links)[:max_links]

def extract_raw_ad_info(driver, ad_url):
    driver.get(ad_url)
    time.sleep(random.uniform(1.2, 2.5)) 
    
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
            except Exception:
                continue

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
            except Exception:
                continue

    except Exception:
        pass
        
    return ad_data

def save_car_to_csv_incremental(filepath, fieldnames, row_dict=None, write_header=False):
    """Write header or append a single row to the CSV file incrementally."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    mode = 'w' if write_header else 'a'
    with open(filepath, mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        if row_dict:
            writer.writerow(row_dict)

def process_single_car_task(car_type):
    """Worker function that performs car scrapes in a separate thread"""
    print(f"\n🚀 [Thread Started] Pipeline for: {car_type}")
    
    driver = setup_driver()
    
    try:
        base_url = f"https://divar.ir/s/iran/car/{car_type}"
        
        # extract first 1000 links
        links = get_ad_links(driver, base_url, car_type, max_links=1000)
        
        output_dir = os.path.join('data', 'cars')
        safe_filename = car_type.replace('/', '_') + '.csv'
        filepath = os.path.join(output_dir, safe_filename)
        
        fieldnames = [
            "brand_model", "year", "mileage", "price", "insurance",
            "gearbox_type", "gearbox_health", "motor_status",
            "body_status", "chassis_status", "url"
        ]
        
        # Initialize the CSV file and write headers
        save_car_to_csv_incremental(filepath, fieldnames, write_header=True)
        
        # processing links in the same browser
        for idx, link in enumerate(links):
            # print progress percentage to avoid terminal clutter
            if idx % 10 == 0 or idx == len(links) - 1:
                print(f"[{car_type}] Processing ad {idx + 1}/{len(links)} ...", end="\r")
                
            raw_info = extract_raw_ad_info(driver, link)
            save_car_to_csv_incremental(filepath, fieldnames, row_dict=raw_info, write_header=False)
            
        print(f"\n[✔] [{car_type}] Data exported successfully to {filepath}")
        
    except Exception as e:
        print(f"\n[!] [{car_type}] Error occurred: {e}")
    finally:
        driver.quit()
        print(f"\n🏁 [{car_type}] WebDriver session terminated.")
        
    return car_type

def main():
    car_types = [
        "pride/131",
        "quick",
        "peugeot/206",
        "peugeot/pars",
        "dena",
        "renault/tondar-90",
        "mvm/x22",
        "jac/s5",
        "kia/cerato",
        "hyundai/santafe-ix45"
    ]
    
    MAX_CONCURRENT_CARS = 2
    
    print(f"🌟 Starting Parallel Scraping with {MAX_CONCURRENT_CARS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CARS) as executor:
        futures = {executor.submit(process_single_car_task, car): car for car in car_types}
        
        # monitoring completed tasks
        for future in as_completed(futures):
            car = futures[future]
            try:
                result = future.result()
                print(f"✅ Task for {result} fully completed.")
            except Exception as exc:
                print(f"❌ Task for {car} generated an exception: {exc}")

    print("\n🎉 All car types have been processed!")

if __name__ == "__main__":
    main()