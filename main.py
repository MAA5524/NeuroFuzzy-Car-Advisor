"""
NeuroFuzzy Car Advisor - Main Entry Point.
This file serves as the main pipeline coordinator. 
Currently demonstrates the execution of Module 1 (Scraper & Data Preprocessing).
"""

import sys
import logging
from scraper import extract_and_clean_car_data

# Set up logging to show progression steps clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_pipeline")


def run_demo():
    """
    Runs a demonstration of Module 1 using a mock URL.
    """
    logger.info("Initializing NeuroFuzzy Car Advisor pipeline...")
    
    # We use a mock URL to demonstrate extraction without external network/webdriver dependencies.
    # Change this to a real Divar URL once you have Chrome/ChromeDriver configured locally.
    demo_url = "mock://divar.ir/v/dena-plus-1402"
    
    logger.info(f"--- Pipeline Step 1: Extracting and cleaning car data from: {demo_url} ---")
    try:
        car_data = extract_and_clean_car_data(demo_url)
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS: Cleaned Car Specs Dictionary (Module 1 Output):")
        print("=" * 50)
        for key, value in car_data.items():
            print(f"  🔹 {key:<22} : {value} (type: {type(value).__name__})")
        print("=" * 50 + "\n")
        
        logger.info("Pipeline Step 1 completed successfully.")
        logger.info("Next steps (Modules 2 & 3) will process this data to predict fair price and recommend value.")
        
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()
