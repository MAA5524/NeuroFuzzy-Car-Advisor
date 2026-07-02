"""
Data Cleaner and Preprocessing Module for NeuroFuzzy Car Advisor.
Handles digit normalization, price/mileage extraction, and body condition scoring.
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
    Supports both Solar Hijri (e.g. 1400) and Gregorian (e.g. 2021) years.
    """
    if not year_str:
        raise ValueError("Year string is empty or None")
        
    normalized = persian_to_english_digits(str(year_str))
    # Extract only the digits
    digits = re.sub(r"\D", "", normalized)
    if not digits:
        raise ValueError(f"Could not extract numeric year from: '{year_str}'")
        
    return int(digits)

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

def map_body_condition(condition_text: str) -> int:
    """
    Maps the Persian/English body condition description to a score from 0 to 100.
    
    Standard mappings:
    - 'بدون رنگ' or 'بی رنگ' or 'No paint' = 100
    - 'یک لکه رنگ' or 'One spot paint' = 80
    - 'دو لکه رنگ' = 70
    - 'دور رنگ' or 'Painted all around' = 50
    - 'تمام رنگ' or 'Fully painted' = 30
    - 'تصادفی' or 'Accident/Crashed' = 10
    """
    if not condition_text:
        return 70  # Default average score if empty
        
    # Normalize characters (e.g. Arabic Yeh/Keheh to Persian)
    normalized = persian_to_english_digits(str(condition_text)).strip().lower()
    normalized = normalized.replace("ي", "ی").replace("ك", "ک")
    
    # 1. No Paint (100)
    if any(phrase in normalized for phrase in ["بدون رنگ", "بی رنگ", "no paint", "perfect"]):
        return 100
        
    # 2. One Spot Paint (80)
    elif any(phrase in normalized for phrase in ["یک لکه رنگ", "1 لکه رنگ", "one spot paint"]):
        return 80
        
    # 3. Two Spots Paint (70)
    elif any(phrase in normalized for phrase in ["دو لکه رنگ", "2 لکه رنگ", "two spot paint"]):
        return 70
        
    # 4. Painted all around (50)
    elif any(phrase in normalized for phrase in ["دور رنگ", "دوررنگ", "painted all around", "around paint"]):
        return 50
        
    # 5. Fully painted / major paint (30)
    elif any(phrase in normalized for phrase in ["تمام رنگ", "تمامرنگ", "fully painted", "full paint"]):
        return 30
        
    # 6. Accidented / Crashed / Scraped (10)
    elif any(phrase in normalized for phrase in ["تصادفی", "تصادف", "crashed", "accident"]):
        return 10
        
    # Fallback search for other keywords
    else:
        logger.warning(f"Unknown body condition text: '{condition_text}'. Defaulting to 70.")
        return 70
