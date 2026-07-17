import json
import numpy as np
import onnxruntime as ort
import sys
import os

# Standalone Mamdani Fuzzy System translation for quick self-contained verification
def trimf(x, a, b, c):
    if x < a or x > c:
        return 0.0
    if a == b and x <= b:
        return 1.0
    if b == c and x >= b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def map_body_status_to_score(body_status):
    mapping = {
        "سالم و بی‌خط و خش": 100.0,
        "صافکاری بی‌رنگ": 85.0,
        "خط و خش جزیی": 75.0,
        "رنگ‌شدگی در ۱ ناحیه": 65.0,
        "دوررنگ": 55.0,
        "رنگ‌شدگی در ۲ ناحیه": 50.0,
        "رنگ‌شدگی در ۳ ناحیه": 40.0,
        "رنگ‌شدگی در ۴ ناحیه": 30.0,
        "رنگ‌شدگی در ۵ ناحیه": 20.0,
        "رنگ‌شدگی در ۷ ناحیه": 10.0,
        "تمام‌رنگ": 15.0,
        "تصادفی": 5.0,
        "اوراقی": 0.0
    }
    return mapping.get(body_status, 75.0)

def map_mechanics_to_score(gearbox_health, motor_status, chassis_status):
    score = 100.0
    if motor_status == "نیاز به تعمیر":
        score -= 35.0
    elif motor_status == "تعویض شده":
        score -= 15.0
    if gearbox_health == "نیاز به تعمیر جزئی":
        score -= 20.0
    elif gearbox_health == "تعمیر شده":
        score -= 15.0
    if chassis_status == "ضربه‌خورده":
        score -= 40.0
    elif chassis_status == "رنگ‌شده":
        score -= 30.0
    return max(0.0, score)

def get_purchase_label(score):
    if score >= 85.0:
        return "خرید عالی (اکازیون)"
    elif score >= 70.0:
        return "خرید خوب"
    elif score >= 50.0:
        return "خرید معمولی"
    elif score >= 30.0:
        return "ارزش خرید پایین"
    else:
        return "خرید نامناسب (توصیه نمی‌شود)"

def run_fuzzy_evaluation(body_status, gearbox_health, motor_status, chassis_status, car_age, predicted_price, asking_price):
    if predicted_price <= 0:
        return 50.0, "خطا در محاسبه قیمت"
    
    price_diff_percent = ((asking_price - predicted_price) / predicted_price) * 100.0
    price_diff_percent = max(-50.0, min(50.0, price_diff_percent))
    
    body_score = map_body_status_to_score(body_status)
    mechanics_score = map_mechanics_to_score(gearbox_health, motor_status, chassis_status)
    age = max(0.0, min(32.0, car_age))
    
    # Membership values
    pd_very_cheap = trimf(price_diff_percent, -50.0, -50.0, -20.0)
    pd_cheap = trimf(price_diff_percent, -35.0, -20.0, 0.0)
    pd_fair = trimf(price_diff_percent, -15.0, 0.0, 15.0)
    pd_expensive = trimf(price_diff_percent, 0.0, 20.0, 35.0)
    pd_very_expensive = trimf(price_diff_percent, 20.0, 50.0, 50.0)
    
    body_poor = trimf(body_score, 0.0, 0.0, 45.0)
    body_mod = trimf(body_score, 30.0, 55.0, 80.0)
    body_good = trimf(body_score, 65.0, 80.0, 95.0)
    body_exc = trimf(body_score, 85.0, 100.0, 100.0)
    
    mech_poor = trimf(mechanics_score, 0.0, 0.0, 45.0)
    mech_mod = trimf(mechanics_score, 30.0, 55.0, 80.0)
    mech_good = trimf(mechanics_score, 65.0, 80.0, 95.0)
    mech_exc = trimf(mechanics_score, 85.0, 100.0, 100.0)
    
    age_new = trimf(age, 0.0, 0.0, 5.0)
    age_mid = trimf(age, 3.0, 8.0, 15.0)
    age_old = trimf(age, 10.0, 32.0, 32.0)
    
    # Strengths
    s_db = []
    s_bad = []
    s_ord = []
    s_good = []
    s_exc = []
    
    s_exc.append(min(pd_very_cheap, mech_exc, body_exc))
    s_exc.append(min(pd_very_cheap, max(mech_good, body_good)))
    s_good.append(min(pd_very_cheap, mech_mod, body_mod))
    s_ord.append(min(pd_very_cheap, max(mech_poor, body_poor)))
    
    s_exc.append(min(pd_cheap, mech_exc, body_exc))
    s_good.append(min(pd_cheap, mech_good, body_good))
    s_ord.append(min(pd_cheap, mech_mod, body_mod))
    s_bad.append(min(pd_cheap, max(mech_poor, body_poor)))
    
    s_good.append(min(pd_fair, mech_exc, body_exc))
    s_good.append(min(pd_fair, mech_good, body_good))
    s_ord.append(min(pd_fair, max(mech_mod, body_mod)))
    s_bad.append(min(pd_fair, max(mech_poor, body_poor)))
    
    s_ord.append(min(pd_expensive, max(mech_exc, body_exc)))
    s_bad.append(min(pd_expensive, mech_good, body_good))
    s_db.append(min(pd_expensive, max(mech_mod, body_mod)))
    s_db.append(min(pd_expensive, max(mech_poor, body_poor)))
    
    s_db.append(pd_very_expensive)
    s_db.append(min(age_old, max(mech_poor, body_poor)))
    s_good.append(min(age_new, pd_fair, mech_exc, body_exc))
    s_exc.append(min(age_new, pd_cheap, mech_exc, body_exc))
    
    agg_db = max(s_db) if s_db else 0.0
    agg_bad = max(s_bad) if s_bad else 0.0
    agg_ord = max(s_ord) if s_ord else 0.0
    agg_good = max(s_good) if s_good else 0.0
    agg_exc = max(s_exc) if s_exc else 0.0
    
    # Defuzzify
    num_sum = 0.0
    den_sum = 0.0
    for y in range(101):
        y_val = float(y)
        mf_db = trimf(y_val, 0.0, 0.0, 35.0)
        mf_bad = trimf(y_val, 25.0, 45.0, 60.0)
        mf_ord = trimf(y_val, 50.0, 65.0, 75.0)
        mf_good = trimf(y_val, 70.0, 80.0, 90.0)
        mf_exc = trimf(y_val, 85.0, 100.0, 100.0)
        
        u_y = max(
            min(agg_db, mf_db),
            min(agg_bad, mf_bad),
            min(agg_ord, mf_ord),
            min(agg_good, mf_good),
            min(agg_exc, mf_exc)
        )
        num_sum += y_val * u_y
        den_sum += u_y
        
    score = 50.0 if den_sum == 0.0 else num_sum / den_sum
    return score, get_purchase_label(score)

def main():
    print("==================================================")
    print("🚗 Car Price Predictor - AI Verification Script 🚗")
    print("==================================================")
    
    # 1. Inputs from UI screenshot
    inputs = {
        'brand': 'peugeot_pars',
        'year': 2020,
        'mileage': 2000,
        'insurance': 12,
        'gearbox_type': 'دنده‌ای',
        'gearbox_health': 'سالم و پلمپ',
        'motor_status': 'سالم',
        'body_status': 'سالم و بی‌خط و خش',
        'chassis_status': 'سالم و پلمپ'
    }
    asking_price = 1000000000.0 # 1 Billion Tomans
    
    print("\n[1] UI inputs:")
    for k, v in inputs.items():
        print(f"  {k:15}: {v}")
    print(f"  {'asking_price':15}: {asking_price:,} Tomans")

    # 2. Load Config
    config_path = 'mlp_model/saved_models/preprocessing_config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    reference_year = config.get('reference_year', 2026)
    car_age = float(reference_year - inputs['year'])
    yearly_mileage = inputs['mileage'] / (car_age + 1.0)
    has_insurance = 1.0 if inputs['insurance'] > 0 else 0.0
    
    print(f"\n[2] Engineered Features:")
    print(f"  car_age        : {car_age}")
    print(f"  yearly_mileage : {yearly_mileage:.4f}")
    print(f"  has_insurance  : {has_insurance}")

    # 3. Manual Preprocessing (Standardization + One-Hot Encoding)
    raw_numerics = [inputs['year'], inputs['mileage'], car_age, yearly_mileage, has_insurance]
    standardized_numerics = []
    means = config['standard_scaler']['mean']
    scales = config['standard_scaler']['scale']
    for i, val in enumerate(raw_numerics):
        std_val = (val - means[i]) / scales[i]
        standardized_numerics.append(std_val)
        
    print(f"\n[3] Standardized Numerical Vector:")
    num_names = ['year', 'mileage', 'car_age', 'yearly_mileage', 'has_insurance']
    for name, val in zip(num_names, standardized_numerics):
        print(f"  {name:15}: {val:.6f}")
        
    one_hot_vector = []
    cat_values = [
        inputs['gearbox_type'],
        inputs['gearbox_health'],
        inputs['motor_status'],
        inputs['body_status'],
        inputs['chassis_status'],
        inputs['brand']
    ]
    
    print("\n[4] One-Hot Encoded Categorical Features:")
    for i, val in enumerate(cat_values):
        field_name = config['categorical_features'][i]
        cats = config['one_hot_categories'][i]
        encoding = [0.0] * len(cats)
        if val in cats:
            idx = cats.index(val)
            encoding[idx] = 1.0
        one_hot_vector.extend(encoding)
        print(f"  {field_name:15}: {encoding} (selected: '{val}')")

    flat_vector = np.array([standardized_numerics + one_hot_vector], dtype=np.float32)
    print(f"\n[5] Concat Output Vector (Shape: {flat_vector.shape}):")
    print(f"  {flat_vector[0].tolist()}")

    # 4. ONNX Inference
    pytorch_model_path = 'mlp_model/saved_models/pytorch_mlp.onnx'
    sklearn_model_path = 'mlp_model/saved_models/sklearn_mlp.onnx'
    
    predicted_toman_pt = None
    predicted_toman_sk = None

    print(f"\n[6] Model Execution:")
    
    # Run PyTorch Model
    try:
        session_pt = ort.InferenceSession(pytorch_model_path)
        outputs_pt = session_pt.run(None, {'input': flat_vector})
        scaled_price_pt = outputs_pt[0][0][0]
        
        target_mean = config['target_scaler']['mean'][0]
        target_scale = config['target_scaler']['scale'][0]
        predicted_toman_pt = (scaled_price_pt * target_scale) + target_mean
        print("  ✅ PyTorch model executed successfully")
    except Exception as e:
        print(f"  ❌ PyTorch model failed: {e}")

    # Run Sklearn Model
    try:
        session_sk = ort.InferenceSession(sklearn_model_path)
        ort_inputs = {
            'year': np.array([[inputs['year']]], dtype=np.int64),
            'mileage': np.array([[inputs['mileage']]], dtype=np.int64),
            'insurance': np.array([[inputs['insurance']]], dtype=np.int64),
            'car_age': np.array([[car_age]], dtype=np.float64),
            'yearly_mileage': np.array([[yearly_mileage]], dtype=np.float64),
            'has_insurance': np.array([[has_insurance]], dtype=np.float64),
            'gearbox_type': np.array([[inputs['gearbox_type']]], dtype=object),
            'gearbox_health': np.array([[inputs['gearbox_health']]], dtype=object),
            'motor_status': np.array([[inputs['motor_status']]], dtype=object),
            'body_status': np.array([[inputs['body_status']]], dtype=object),
            'chassis_status': np.array([[inputs['chassis_status']]], dtype=object),
            'brand': np.array([[inputs['brand']]], dtype=object)
        }
        # Cast inputs based on expected ONNX types
        for i in session_sk.get_inputs():
            if i.type == 'tensor(int64)' and ort_inputs[i.name].dtype != np.int64:
                ort_inputs[i.name] = ort_inputs[i.name].astype(np.int64)
            elif i.type == 'tensor(float)' and ort_inputs[i.name].dtype != np.float32:
                ort_inputs[i.name] = ort_inputs[i.name].astype(np.float32)
            elif i.type == 'tensor(double)' and ort_inputs[i.name].dtype != np.float64:
                ort_inputs[i.name] = ort_inputs[i.name].astype(np.float64)

        outputs_sk = session_sk.run(None, ort_inputs)
        scaled_price_sk = outputs_sk[0][0][0]
        
        predicted_toman_sk = (scaled_price_sk * target_scale) + target_mean
        print("  ✅ Sklearn model executed successfully")
    except Exception as e:
        print(f"  ❌ Sklearn model failed: {e}")

    if predicted_toman_pt is None and predicted_toman_sk is None:
        print("  Error: Both models failed to predict.")
        return

    print(f"\n[7] Prediction Result:")
    if predicted_toman_pt is not None:
        print(f"  PyTorch Predicted Price : {predicted_toman_pt:,.2f} Tomans")
    if predicted_toman_sk is not None:
        print(f"  Sklearn Predicted Price : {predicted_toman_sk:,.2f} Tomans")
        
    primary_prediction = predicted_toman_pt if predicted_toman_pt is not None else predicted_toman_sk
        
    try:
        # 5. Fuzzy Logic Purchase Advice
        score, label = run_fuzzy_evaluation(
            body_status=inputs['body_status'],
            gearbox_health=inputs['gearbox_health'],
            motor_status=inputs['motor_status'],
            chassis_status=inputs['chassis_status'],
            car_age=car_age,
            predicted_price=primary_prediction,
            asking_price=asking_price
        )
        
        price_diff = ((asking_price - primary_prediction) / primary_prediction) * 100.0
        
        print(f"\n[8] Fuzzy Purchase Recommendation:")
        print(f"  Price Difference  : {price_diff:+.2f}%")
        print(f"  Fuzzy Score       : {score:.2f} / 100")
        print(f"  Fuzzy Label       : {label}")
        
    except Exception as e:
        print(f"  Error running fuzzy evaluation: {e}")

if __name__ == "__main__":
    main()
