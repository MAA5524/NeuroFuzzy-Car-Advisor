# 🚗 NeuroFuzzy Car Advisor

**A Smart Car Purchasing Recommendation System integrating Multilayer Perceptron (MLP) Regression and Fuzzy Decision Making.**

This project was developed as a final assignment for the **Computational Intelligence** course. It aims to help car buyers make informed decisions by combining real-world data extraction, neural network price prediction, and human-like fuzzy logic evaluation.

---

## 📌 Project Overview

When buying a used car, buyers often struggle to determine if the seller's asking price is fair considering the car's condition. This system automates the evaluation process using a three-step pipeline:

1. **Web Scraping:** Automatically extracts car listings (year, mileage, body condition, and seller's price) from online platforms (e.g., Bama or Divar).
2. **Fair Price Prediction (MLP):** A trained Multilayer Perceptron Neural Network evaluates the car's technical specs and predicts its actual market value.
3. **Fuzzy Decision Making:** A Mamdani Fuzzy Inference System compares the *Seller's Price* with the *MLP Predicted Price*, factors in the *Body Condition*, and outputs a final "Purchase Value" score (e.g., "Do Not Buy", "Fair Deal", "Excellent/Bargain").

---

## 🏗️ System Architecture & Division of Labor

The project is highly modular and divided into three main components:

### 1. Web Scraper Module (`/scraper`)
* **Role:** Gathers raw data from car advertisement websites.
* **Outputs:** A structured dataset including car features and the seller's asking price.
* **Tools:** `Selenium`, `BeautifulSoup`, `Requests`.

### 2. Neural Network Module (`/mlp_model`)
* **Role:** Acts as the "Expert Appraiser". It takes historical/cleaned data to train an MLP Regressor.
* **Inputs:** Car specs (Age, Mileage, etc.).
* **Outputs:** The estimated fair price of the vehicle.
* **Tools:** `scikit-learn`, `pandas`.

### 3. Fuzzy Logic Module (`/fuzzy_system`)
* **Role:** Acts as the "Decision Maker". 
* **Inputs:** 
  - Price Difference (Seller Price vs. MLP Predicted Price).
  - Physical Health / Body Condition (extracted by the scraper).
* **Outputs:** A logical recommendation percentage or label (Purchase Value).
* **Tools:** `scikit-fuzzy`.

---

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Machine Learning:** `scikit-learn`, `numpy`, `pandas`
* **Fuzzy Logic:** `scikit-fuzzy`
* **Web Scraping:** `selenium`, `beautifulsoup4`

---

## 📂 Project Structure

```text
NeuroFuzzy-Car-Advisor/
├── data/                       # Raw and cleaned datasets
├── scraper/                    # Web scraping scripts
├── mlp_model/                  # Neural network training & prediction
├── fuzzy_system/               # Fuzzy logic rules and inference
├── main.py                     # Main pipeline executing the whole process
└── requirements.txt            # Project dependencies

```

---

## 🚀 Setup & Installation

**1. Clone the repository:**

```bash
git clone git@github.com:MAA5524/NeuroFuzzy-Car-Advisor.git
cd NeuroFuzzy-Car-Advisor

```

**2. Create a virtual environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install dependencies:**

```bash
pip install -r requirements.txt

```

*(Note: Depending on the scraper setup, you may also need a compatible web driver like ChromeDriver for Selenium).*

---

## 💻 How to Run

To run the complete pipeline (Scrape -> Predict -> Evaluate), simply execute the main file:

```bash
python main.py

```
