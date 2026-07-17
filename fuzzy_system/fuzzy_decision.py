# fuzzy_decision.py
import sys
import os
from skfuzzy import control as ctrl

# Handle imports dynamically for both direct execution and package execution
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fuzzy_system.fuzzy_rules import define_rules_and_variables
else:
    from .fuzzy_rules import define_rules_and_variables


def build_fuzzy_simulator():
    """Builds and returns the fuzzy control system simulator."""
    rules = define_rules_and_variables()
    purchase_control = ctrl.ControlSystem(rules)
    purchase_simulator = ctrl.ControlSystemSimulation(purchase_control)
    return purchase_simulator


def map_body_status_to_score(body_status):
    """
    Maps the Persian string representing body condition to a numerical score [0, 100].
    """
    if not isinstance(body_status, str):
        return 75.0  # Fallback to a neutral score (moderate/good)

    body_status = body_status.strip()

    # Exact mapping based on dataset distribution and typical car valuation conventions
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
    """
    Maps technical state variables (gearbox, motor, chassis) to a combined score [0, 100].
    """
    score = 100.0

    # Motor Status deductions
    if motor_status == "نیاز به تعمیر":
        score -= 35.0
    elif motor_status == "تعویض شده":
        score -= 15.0

    # Gearbox Health deductions
    if gearbox_health == "نیاز به تعمیر جزئی":
        score -= 20.0
    elif gearbox_health == "تعمیر شده":
        score -= 15.0

    # Chassis Status deductions
    if chassis_status == "ضربه‌خورده":
        score -= 40.0
    elif chassis_status == "رنگ‌شده":
        score -= 30.0

    return max(0.0, score)


def get_purchase_label(score):
    """
    Translates the numerical defuzzified purchase value to a descriptive Persian label.
    """
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


def evaluate_car_purchase(features_dict, predicted_price, asking_price):
    """
    Evaluates the deal purchase score and label using fuzzy logic.
    
    Args:
        features_dict (dict): Dictionary containing car features:
            - 'body_status'
            - 'gearbox_health'
            - 'motor_status'
            - 'chassis_status'
            - 'car_age' or 'year'
        predicted_price (float): MLP predicted market price.
        asking_price (float): Seller's asking price.
        
    Returns:
        dict: Containing 'purchase_score' [0-100] and 'label' (Persian string).
    """
    try:
        # 1. Compute price difference percentage
        if predicted_price <= 0:
            raise ValueError("Predicted price must be greater than zero.")
        
        price_diff_percent = ((asking_price - predicted_price) / predicted_price) * 100
        # Restrict price_diff_percent to [-50, 50] range of the universe
        price_diff_percent = max(min(price_diff_percent, 50.0), -50.0)

        # 2. Extract and map body/mechanical specs
        body_status = features_dict.get('body_status', 'سالم و بی‌خط و خش')
        body_score = map_body_status_to_score(body_status)

        gearbox_health = features_dict.get('gearbox_health', 'سالم و پلمپ')
        motor_status = features_dict.get('motor_status', 'سالم')
        chassis_status = features_dict.get('chassis_status', 'سالم و پلمپ')
        mechanics_score = map_mechanics_to_score(gearbox_health, motor_status, chassis_status)

        # 3. Extract and map car age
        age = features_dict.get('car_age')
        if age is None:
            # Fallback calculation if year is present
            year = features_dict.get('year')
            if year is not None:
                age = max(0, 2026 - int(year))
            else:
                age = 5.0  # Default fallback
        else:
            age = float(age)
        
        # Restrict age to [0, 32]
        age = max(min(age, 32.0), 0.0)

        # 4. Initialize and run simulator
        simulator = build_fuzzy_simulator()
        simulator.input['price_diff'] = price_diff_percent
        simulator.input['body_condition'] = body_score
        simulator.input['mechanics_condition'] = mechanics_score
        simulator.input['car_age'] = age

        simulator.compute()

        # 5. Extract output and construct result
        purchase_score = float(simulator.output['purchase_value'])
        label = get_purchase_label(purchase_score)

        return {
            "price_diff_percent": round(price_diff_percent, 2),
            "body_score": body_score,
            "mechanics_score": mechanics_score,
            "car_age": age,
            "purchase_score": round(purchase_score, 2),
            "label": label
        }

    except Exception as e:
        print(f"Error in evaluating deal: {str(e)}")
        return {
            "price_diff_percent": 0.0,
            "body_score": 50.0,
            "mechanics_score": 50.0,
            "car_age": 5.0,
            "purchase_score": 50.0,
            "label": f"خطا در محاسبه: {str(e)}"
        }


# Quick backward compatible function
def evaluate_deal_value(price_diff_percent, body_score):
    """
    Backward-compatible wrapper evaluation function.
    """
    sim = build_fuzzy_simulator()
    sim.input['price_diff'] = max(min(price_diff_percent, 50.0), -50.0)
    sim.input['body_condition'] = max(min(body_score, 100.0), 0.0)
    sim.input['mechanics_condition'] = 100.0  # Perfect by default
    sim.input['car_age'] = 3.0  # New/mid age by default
    sim.compute()
    return round(float(sim.output['purchase_value']), 2)


if __name__ == "__main__":
    # Test cases
    print("Testing fuzzy system decision evaluations:")
    
    # Test Case 1: Cheap price (-15%), Excellent body (100) and Mechanics (100), New car (2 years old)
    test_car_1 = {
        'year': 2024,
        'car_age': 2,
        'body_status': 'سالم و بی‌خط و خش',
        'gearbox_health': 'سالم و پلمپ',
        'motor_status': 'سالم',
        'chassis_status': 'سالم و پلمپ'
    }
    res_1 = evaluate_car_purchase(test_car_1, predicted_price=1000000000, asking_price=850000000)
    print("\nTest Case 1 (Good Deal):")
    print(res_1)

    # Test Case 2: Expensive price (+20%), Poor body (دوررنگ), Poor chassis (ضربه‌خورده)
    test_car_2 = {
        'year': 2015,
        'car_age': 11,
        'body_status': 'دوررنگ',
        'gearbox_health': 'سالم و پلمپ',
        'motor_status': 'سالم',
        'chassis_status': 'ضربه‌خورده'
    }
    res_2 = evaluate_car_purchase(test_car_2, predicted_price=500000000, asking_price=600000000)
    print("\nTest Case 2 (Bad Deal):")
    print(res_2)