"""
Data Cleaner and Preprocessing Module for NeuroFuzzy Car Advisor.
Handles digit normalization, price/mileage extraction, body condition scoring,
and overall condition score calculation.
"""

import re
import logging

logger = logging.getLogger(__name__)

def persian_to_english_digits(text: str) -> str:
    """
    Converts Persian (۰-۹) and Arabic (٠-٩) numerals in a string to standard English digits (0-9).
    """
    if not isinstance(text, str):
        return ""
    
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    
    # Map Persian to English
    translation_table = str.maketrans(
        persian_digits + arabic_digits, 
        english_digits + english_digits
    )
    return text.translate(translation_table)

def clean_year(year_str: str) -> int:
    """
    Cleans and extracts the car manufacturing year as an integer.
    Supports both Solar Hijri (e.g. 1399) and Gregorian (e.g. 2021) years.
    Prioritizes Solar Hijri years if both are present in the string.
    """
    if not year_str:
        raise ValueError("Year string is empty or None")
        
    normalized = persian_to_english_digits(str(year_str))
    
    # Find all 4-digit numbers
    years = re.findall(r"\b\d{4}\b", normalized)
    if not years:
        # Fallback to any digits
        digits = re.sub(r"\D", "", normalized)
        if not digits:
            raise ValueError(f"Could not extract numeric year from: '{year_str}'")
        return int(digits)
    
    # Check if any found 4-digit year is in the Solar Hijri range (1300 to 1450)
    for y in years:
        val = int(y)
        if 1300 <= val <= 1450:
            return val
            
    # Return the first 4-digit year found
    return int(years[0])

def clean_mileage(mileage_str: str) -> int:
    """
    Cleans mileage strings and converts them to standard integers.
    Handles 'صفر' (Zero), 'کارتکس' (New), and 'کیلومتر' units.
    """
    if not mileage_str:
        return 0
        
    normalized = persian_to_english_digits(str(mileage_str)).strip().lower()
    
    # Check for keywords indicating a brand new car (zero mileage)
    zero_keywords = ["صفر", "zero", "کارتکس", "کارنکرده", "نو"]
    if normalized == "0" or any(kw in normalized for kw in zero_keywords):
        return 0
        
    # Extract only the digits
    digits = re.sub(r"\D", "", normalized)
    if not digits:
        logger.warning(f"Could not find numeric digits in mileage: '{mileage_str}'. Defaulting to 0.")
        return 0
        
    return int(digits)

def clean_seller_price(price_str: str) -> int:
    """
    Cleans seller's price (removes commas, spaces, and 'تومان' or 'Toman')
    and converts it to a standard integer.
    """
    if not price_str:
        raise ValueError("Price string is empty or None")
        
    normalized = persian_to_english_digits(str(price_str)).strip()
    
    # Check if the price is negotiable / agreement-based (توافقی)
    if "توافقی" in normalized or "agreement" in normalized.lower():
        # Raise ValueError because MLP requires a real price to operate
        raise ValueError("Price is negotiable (توافقی) and cannot be converted to a numeric price.")
        
    # Remove commas, spaces, and currencies
    cleaned = normalized.replace(",", "").replace(" ", "")
    cleaned = re.sub(r"تومان|toman|Toman|ریال|rial|Rial", "", cleaned)
    
    # Extract only digits
    digits = re.sub(r"\D", "", cleaned)
    if not digits:
        raise ValueError(f"Could not extract numeric price from: '{price_str}'")
        
    return int(digits)

def clean_insurance(insurance_str: str) -> int:
    """
    Cleans insurance string and extracts the remaining months of insurance.
    E.g. '۶ ماه' -> 6, 'بدون بیمه' -> 0.
    """
    if not insurance_str:
        return 0
    normalized = persian_to_english_digits(str(insurance_str)).strip().lower()
    if any(kw in normalized for kw in ["بدون", "فاقد", "no"]):
        return 0
    digits = re.sub(r"\D", "", normalized)
    if not digits:
        return 0
    return int(digits)

def map_body_condition(condition_text: str) -> int:
    """
    Maps the Persian/English body condition description to a score from 0 to 100.
    
    Standard mappings:
    - 'بدون رنگ' or 'بی رنگ' or 'No paint' = 100
    - 'خط و خش جزیی' = 90
    - 'یک لکه رنگ' or 'رنگ‌شدگی در ۱ ناحیه' = 80
    - 'دو لکه رنگ' or 'رنگ‌شدگی در ۲ ناحیه' = 70
    - 'سه لکه رنگ' or 'رنگ‌شدگی در ۳ ناحیه' = 60
    - 'دور رنگ' or 'Painted all around' = 50
    - 'تمام رنگ' or 'Fully painted' = 30
    - 'تصادفی' or 'Accident/Crashed' = 10
    """
    if not condition_text:
        return 70  # Default average score if empty
        
    # Normalize characters
    normalized = persian_to_english_digits(str(condition_text)).strip().lower()
    normalized = normalized.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    
    # 1. No Paint (100)
    if any(phrase in normalized for phrase in ["بدون رنگ", "بی رنگ", "no paint", "perfect", "سالم و پلمپ"]):
        return 100
        
    # 2. Minor scratches / Line and scratch (90)
    elif any(phrase in normalized for phrase in ["خط و خش جزیی", "خط و خش جزئی", "خط و خش", "صافکاری بدون رنگ"]):
        return 90
        
    # 3. One Spot Paint (80)
    elif any(phrase in normalized for phrase in ["یک لکه رنگ", "1 لکه رنگ", "one spot paint", "رنگ شدگی در 1 ناحیه", "رنگ شدگی در یک ناحیه"]):
        return 80
        
    # 4. Two Spots Paint (70)
    elif any(phrase in normalized for phrase in ["دو لکه رنگ", "2 لکه رنگ", "two spot paint", "رنگ شدگی در 2 ناحیه", "رنگ شدگی در دو ناحیه"]):
        return 70
        
    # 5. Three Spots Paint (60)
    elif any(phrase in normalized for phrase in ["سه لکه رنگ", "3 لکه رنگ", "three spot paint", "رنگ شدگی در 3 ناحیه", "رنگ شدگی در سه ناحیه"]):
        return 60
        
    # 6. Painted all around (50)
    elif any(phrase in normalized for phrase in ["دور رنگ", "دوررنگ", "painted all around", "around paint", "رنگ شدگی در چند ناحیه"]):
        return 50
        
    # 7. Fully painted / major paint (30)
    elif any(phrase in normalized for phrase in ["تمام رنگ", "تمامرنگ", "fully painted", "full paint"]):
        return 30
        
    # 8. Accidented / Crashed / Scraped (10)
    elif any(phrase in normalized for phrase in ["تصادفی", "تصادف", "crashed", "accident"]):
        return 10
        
    # Fallback search for other keywords
    else:
        logger.warning(f"Unknown body condition text: '{condition_text}'. Defaulting to 70.")
        return 70

def calculate_condition_score(body_score: int, motor_status: str, gearbox_health: str, chassis_status: str) -> float:
    """
    Calculates an overall condition score (0-100) based on body condition score
    and status of motor, gearbox, and chassis.
    """
    score = float(body_score)
    
    # Motor status deductions
    if motor_status:
        motor_norm = persian_to_english_digits(str(motor_status)).strip()
        if "تعمیر" in motor_norm or "خراب" in motor_norm:
            score -= 25
        elif "تعویض" in motor_norm:
            score -= 15
        elif "بخار" in motor_norm:
            score -= 10
            
    # Gearbox health deductions
    if gearbox_health:
        gear_norm = persian_to_english_digits(str(gearbox_health)).strip()
        if "تعمیر" in gear_norm or "خراب" in gear_norm:
            score -= 15
        elif "تقه" in gear_norm or "مشکل" in gear_norm:
            score -= 10
            
    # Chassis status deductions
    if chassis_status:
        chassis_norm = persian_to_english_digits(str(chassis_status)).strip()
        if "ضربه" in chassis_norm or "خوردگی" in chassis_norm:
            score -= 20
        elif "ترک" in chassis_norm or "جوش" in chassis_norm:
            score -= 25
        elif "رنگ" in chassis_norm:
            score -= 10
            
    # Ensure score stays in 0-100 range
    return max(0.0, min(100.0, score))

def clean_car_record(raw_data: dict) -> dict:
    """
    Cleans a single raw car record (dictionary) and computes its condition score.
    """
    cleaned = dict(raw_data)
    
    # 1. Clean Year
    try:
        cleaned["year"] = clean_year(raw_data.get("year", ""))
    except Exception as e:
        logger.warning(f"Error cleaning year: {e}. Defaulting to None.")
        cleaned["year"] = None
        
    # 2. Clean Mileage
    try:
        cleaned["mileage"] = clean_mileage(raw_data.get("mileage", ""))
    except Exception as e:
        logger.warning(f"Error cleaning mileage: {e}. Defaulting to 0.")
        cleaned["mileage"] = 0
        
    # 3. Clean Price
    try:
        cleaned["price"] = clean_seller_price(raw_data.get("price", ""))
    except Exception as e:
        logger.warning(f"Error cleaning price: {e}. Defaulting to None.")
        cleaned["price"] = None
        
    # 4. Clean Insurance
    cleaned["insurance"] = clean_insurance(raw_data.get("insurance", ""))
    
    # 5. Map Body Condition
    body_score = map_body_condition(raw_data.get("body_status", ""))
    
    # 6. Calculate overall condition score
    cleaned["condition_score"] = calculate_condition_score(
        body_score=body_score,
        motor_status=raw_data.get("motor_status", ""),
        gearbox_health=raw_data.get("gearbox_health", ""),
        chassis_status=raw_data.get("chassis_status", "")
    )
    
    return cleaned

def extract_and_clean_car_data(ad_url: str) -> dict:
    """
    Extracts raw car data from the given Divar ad URL and cleans it.
    If the URL is a mock URL (starts with 'mock://'), returns clean mock data.
    """
    if ad_url.startswith("mock://"):
        # Return mock data for demonstration in main.py
        mock_data = {
            "brand_model": "دنا پلاس Dena Plus",
            "year": 1402,
            "mileage": 15000,
            "price": 850000000,
            "insurance": 6,
            "gearbox_type": "دنده‌ای",
            "gearbox_health": "سالم و پلمپ",
            "motor_status": "سالم",
            "body_status": "بدون رنگ",
            "chassis_status": "سالم و پلمپ",
            "condition_score": 100.0,
            "url": ad_url
        }
        return mock_data
        
    # Otherwise, perform actual scraping
    from .scraper import setup_driver, extract_raw_ad_info
    
    driver = setup_driver()
    try:
        raw_data = extract_raw_ad_info(driver, ad_url)
        cleaned_data = clean_car_record(raw_data)
        return cleaned_data
    finally:
        driver.quit()

def clean_csv(input_path: str, output_path: str) -> None:
    """
    Reads raw scraped data from input_path, cleans all records,
    and writes the cleaned records to output_path.
    """
    import csv
    
    logger.info(f"Cleaning CSV data from {input_path} to {output_path}...")
    
    try:
        with open(input_path, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            if fieldnames is None:
                raise ValueError("CSV file has no headers")
                
            out_fieldnames = list(fieldnames)
            if "condition_score" not in out_fieldnames:
                out_fieldnames.append("condition_score")
                
            rows_to_write = []
            for row in reader:
                if not any(row.values()):
                    continue
                try:
                    cleaned_row = clean_car_record(row)
                    rows_to_write.append(cleaned_row)
                except Exception as e:
                    logger.error(f"Error cleaning row {row.get('url', '')}: {e}")
                    
        with open(output_path, mode="w", newline="", encoding="utf-8-sig") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)
            
        logger.info(f"Successfully cleaned {len(rows_to_write)} records and saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to clean CSV: {e}")
        raise

def merge_csv_files(input_dir: str, output_path: str) -> None:
    """
    Reads all CSV files in input_dir, merges them, and writes to output_path.
    """
    import os
    import csv
    import glob
    
    logger.info(f"Merging CSV files from {input_dir} into {output_path}...")
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return
        
    all_rows = []
    fieldnames = None
    
    for file_path in csv_files:
        with open(file_path, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)
                
    if not fieldnames:
        logger.warning("Could not determine headers for merged CSV.")
        return
        
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    logger.info(f"Successfully merged {len(csv_files)} files into {output_path} with {len(all_rows)} rows.")

if __name__ == "__main__":
    import os
    # Setup simple logging for CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # We resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cars_dir = os.path.join(project_root, "data", "cars")
    raw_data_file = os.path.join(project_root, "data", "raw_data.csv")
    cleaned_data_file = os.path.join(project_root, "data", "cleaned_data.csv")
    
    if os.path.exists(cars_dir):
        merge_csv_files(cars_dir, raw_data_file)
        if os.path.exists(raw_data_file):
            clean_csv(raw_data_file, cleaned_data_file)
    else:
        logger.error(f"Input directory not found at: {cars_dir}")

