# fuzzy_rules.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def define_rules_and_variables():
    # 1. Universes of Discourse
    universe_price_diff = np.arange(-50, 51, 1)
    universe_body = np.arange(0, 101, 1)
    universe_mechanics = np.arange(0, 101, 1)
    universe_age = np.arange(0, 33, 1)
    universe_purchase = np.arange(0, 101, 1)

    # 2. Antecedents and Consequent
    price_diff = ctrl.Antecedent(universe_price_diff, 'price_diff')
    body_condition = ctrl.Antecedent(universe_body, 'body_condition')
    mechanics_condition = ctrl.Antecedent(universe_mechanics, 'mechanics_condition')
    car_age = ctrl.Antecedent(universe_age, 'car_age')
    purchase_value = ctrl.Consequent(universe_purchase, 'purchase_value')

    # 3. Membership Functions for price_diff
    price_diff['very_cheap'] = fuzz.trimf(universe_price_diff, [-50, -50, -20])
    price_diff['cheap'] = fuzz.trimf(universe_price_diff, [-35, -20, 0])
    price_diff['fair'] = fuzz.trimf(universe_price_diff, [-15, 0, 15])
    price_diff['expensive'] = fuzz.trimf(universe_price_diff, [0, 20, 35])
    price_diff['very_expensive'] = fuzz.trimf(universe_price_diff, [20, 50, 50])

    # 4. Membership Functions for body_condition
    body_condition['poor'] = fuzz.trimf(universe_body, [0, 0, 45])
    body_condition['moderate'] = fuzz.trimf(universe_body, [30, 55, 80])
    body_condition['good'] = fuzz.trimf(universe_body, [65, 80, 95])
    body_condition['excellent'] = fuzz.trimf(universe_body, [85, 100, 100])

    # 5. Membership Functions for mechanics_condition
    mechanics_condition['poor'] = fuzz.trimf(universe_mechanics, [0, 0, 45])
    mechanics_condition['moderate'] = fuzz.trimf(universe_mechanics, [30, 55, 80])
    mechanics_condition['good'] = fuzz.trimf(universe_mechanics, [65, 80, 95])
    mechanics_condition['excellent'] = fuzz.trimf(universe_mechanics, [85, 100, 100])

    # 6. Membership Functions for car_age
    car_age['new'] = fuzz.trimf(universe_age, [0, 0, 5])
    car_age['mid'] = fuzz.trimf(universe_age, [3, 8, 15])
    car_age['old'] = fuzz.trimf(universe_age, [10, 32, 32])

    # 7. Membership Functions for purchase_value
    purchase_value['dont_buy'] = fuzz.trimf(universe_purchase, [0, 0, 35])
    purchase_value['bad'] = fuzz.trimf(universe_purchase, [25, 45, 60])
    purchase_value['ordinary'] = fuzz.trimf(universe_purchase, [50, 65, 75])
    purchase_value['good'] = fuzz.trimf(universe_purchase, [70, 80, 90])
    purchase_value['excellent'] = fuzz.trimf(universe_purchase, [85, 100, 100])

    # 8. Fuzzy Rules
    rules = [
        # -- Very Cheap Price --
        ctrl.Rule(price_diff['very_cheap'] & mechanics_condition['excellent'] & body_condition['excellent'], purchase_value['excellent']),
        ctrl.Rule(price_diff['very_cheap'] & (mechanics_condition['good'] | body_condition['good']), purchase_value['excellent']),
        ctrl.Rule(price_diff['very_cheap'] & mechanics_condition['moderate'] & body_condition['moderate'], purchase_value['good']),
        ctrl.Rule(price_diff['very_cheap'] & (mechanics_condition['poor'] | body_condition['poor']), purchase_value['ordinary']),

        # -- Cheap Price --
        ctrl.Rule(price_diff['cheap'] & mechanics_condition['excellent'] & body_condition['excellent'], purchase_value['excellent']),
        ctrl.Rule(price_diff['cheap'] & mechanics_condition['good'] & body_condition['good'], purchase_value['good']),
        ctrl.Rule(price_diff['cheap'] & mechanics_condition['moderate'] & body_condition['moderate'], purchase_value['ordinary']),
        ctrl.Rule(price_diff['cheap'] & (mechanics_condition['poor'] | body_condition['poor']), purchase_value['bad']),

        # -- Fair Price --
        ctrl.Rule(price_diff['fair'] & mechanics_condition['excellent'] & body_condition['excellent'], purchase_value['good']),
        ctrl.Rule(price_diff['fair'] & mechanics_condition['good'] & body_condition['good'], purchase_value['good']),
        ctrl.Rule(price_diff['fair'] & (mechanics_condition['moderate'] | body_condition['moderate']), purchase_value['ordinary']),
        ctrl.Rule(price_diff['fair'] & (mechanics_condition['poor'] | body_condition['poor']), purchase_value['bad']),

        # -- Expensive Price --
        ctrl.Rule(price_diff['expensive'] & (mechanics_condition['excellent'] | body_condition['excellent']), purchase_value['ordinary']),
        ctrl.Rule(price_diff['expensive'] & mechanics_condition['good'] & body_condition['good'], purchase_value['bad']),
        ctrl.Rule(price_diff['expensive'] & (mechanics_condition['moderate'] | body_condition['moderate']), purchase_value['dont_buy']),
        ctrl.Rule(price_diff['expensive'] & (mechanics_condition['poor'] | body_condition['poor']), purchase_value['dont_buy']),

        # -- Very Expensive Price --
        ctrl.Rule(price_diff['very_expensive'], purchase_value['dont_buy']),

        # -- Age specific conditions --
        ctrl.Rule(car_age['old'] & (mechanics_condition['poor'] | body_condition['poor']), purchase_value['dont_buy']),
        ctrl.Rule(car_age['new'] & price_diff['fair'] & mechanics_condition['excellent'] & body_condition['excellent'], purchase_value['good']),
        ctrl.Rule(car_age['new'] & price_diff['cheap'] & mechanics_condition['excellent'] & body_condition['excellent'], purchase_value['excellent'])
    ]

    return rules