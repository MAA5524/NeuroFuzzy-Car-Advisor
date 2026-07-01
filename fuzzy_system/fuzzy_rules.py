# fuzzy_rules.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def define_rules_and_variables():
    # ۱. تعریف دامنه‌ها (Universes)
    universe_price_diff = np.arange(-50, 51, 1)
    universe_body = np.arange(0, 101, 1)
    universe_purchase = np.arange(0, 101, 1)

    # ۲. تعریف متغیرهای زبانی ورودی و خروجی
    price_diff = ctrl.Antecedent(universe_price_diff, 'price_diff')
    body_condition = ctrl.Antecedent(universe_body, 'body_condition')
    purchase_value = ctrl.Consequent(universe_purchase, 'purchase_value')

    # ۳. تعریف توابع عضویت مثلثی برای تفاوت قیمت (price_diff)
    price_diff['cheap'] = fuzz.trimf(universe_price_diff, [-50, -50, 0])
    price_diff['fair'] = fuzz.trimf(universe_price_diff, [-20, 0, 20])
    price_diff['expensive'] = fuzz.trimf(universe_price_diff, [0, 50, 50])

    # ۴. تعریف توابع عضویت مثلثی برای سلامت بدنه (body_condition)
    body_condition['poor'] = fuzz.trimf(universe_body, [0, 0, 40])
    body_condition['average'] = fuzz.trimf(universe_body, [30, 55, 80])
    body_condition['excellent'] = fuzz.trimf(universe_body, [70, 100, 100])

    # ۵. تعریف توابع عضویت مثلثی برای ارزش خرید (purchase_value)
    purchase_value['dont_buy'] = fuzz.trimf(universe_purchase, [0, 0, 40])
    purchase_value['ordinary'] = fuzz.trimf(universe_purchase, [30, 50, 70])
    purchase_value['bargain'] = fuzz.trimf(universe_purchase, [60, 100, 100])

    # ۶. تعریف پایگاه قوانین فازی
    rule1 = ctrl.Rule(price_diff['expensive'] & body_condition['poor'], purchase_value['dont_buy'])
    rule2 = ctrl.Rule(price_diff['cheap'] & body_condition['excellent'], purchase_value['bargain'])
    rule3 = ctrl.Rule(price_diff['fair'] & body_condition['average'], purchase_value['ordinary'])
    rule4 = ctrl.Rule(price_diff['expensive'] & body_condition['excellent'], purchase_value['ordinary'])
    rule5 = ctrl.Rule(price_diff['cheap'] & body_condition['poor'], purchase_value['ordinary'])

    # بازگرداندن قوانین تعریف شده برای استفاده در ماژول تصمیم‌گیری
    return [rule1, rule2, rule3, rule4, rule5]