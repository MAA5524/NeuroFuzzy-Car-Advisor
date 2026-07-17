"""
NeuroFuzzy Car Advisor - Main Entry Point.
This file serves as the main pipeline coordinator. 
Demonstrates the execution of the full end-to-end pipeline:
1. Web Scraping & Data Preprocessing (Module 1)
2. MLP Neural Network Fair Price Prediction (Module 2)
3. Fuzzy Decision Advisor (Module 3)
"""

import sys
import logging
from scraper import extract_and_clean_car_data
from mlp_model.predict_price_pytorch import predict_car_price
from fuzzy_system import evaluate_car_purchase

# Set up logging to show progression steps clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_pipeline")


def run_demo():
    """
    Runs a demonstration of the full NeuroFuzzy pipeline (Scraper -> MLP -> Fuzzy System).
    """
    logger.info("Initializing NeuroFuzzy Car Advisor pipeline...")
    
    # We use a mock URL to demonstrate extraction without external network/webdriver dependencies.
    # Change this to a real Divar URL once you have Chrome/ChromeDriver configured locally.
    demo_url = "mock://divar.ir/v/dena-plus-1402"
    
    try:
        # Step 1: Web Scraper & Preprocessing
        logger.info(f"--- Pipeline Step 1: Extracting and cleaning car data from: {demo_url} ---")
        car_data = extract_and_clean_car_data(demo_url)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Cleaned Car Specs Dictionary (Module 1 Output):")
        print("=" * 60)
        for key, value in car_data.items():
            print(f"  🔹 {key:<22} : {value}")
        print("=" * 60 + "\n")
        
        # Step 2: MLP Fair Price Prediction
        logger.info("--- Pipeline Step 2: Predicting fair price using MLP Neural Network (Module 2) ---")
        predicted_price = predict_car_price(car_data)
        asking_price = car_data["price"]
        
        print("\n" + "=" * 60)
        print("🤖 MLP PRICE ESTIMATOR RESULT:")
        print("=" * 60)
        print(f"  🔹 Asking Price     : {asking_price:,.0f} Tomans")
        print(f"  🔹 MLP Fair Price   : {predicted_price:,.0f} Tomans")
        print("=" * 60 + "\n")
        
        # Step 3: Fuzzy Inference Purchase Evaluation
        logger.info("--- Pipeline Step 3: Evaluating value using Mamdani Fuzzy System (Module 3) ---")
        evaluation = evaluate_car_purchase(car_data, predicted_price, asking_price)
        
        print("\n" + "=" * 60)
        print("⚖️ FUZZY DECISION MAKER RECOMMENDATION:")
        print("=" * 60)
        print(f"  🔹 Price Difference : {evaluation['price_diff_percent']}%")
        print(f"  🔹 Body Score       : {evaluation['body_score']}/100")
        print(f"  🔹 Mechanics Score  : {evaluation['mechanics_score']}/100")
        print(f"  🔹 Car Age          : {evaluation['car_age']} years")
        print(f"  🔹 Purchase Score   : {evaluation['purchase_score']}/100")
        print(f"  🔹 Recommendation   : {evaluation['label']}")
        print("=" * 60 + "\n")
        
        logger.info("Full pipeline executed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()
