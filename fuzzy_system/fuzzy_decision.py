# fuzzy_decision.py
from skfuzzy import control as ctrl
# ایمپورت کردن تابع تعریف قوانین از فایل مجاور
from fuzzy_rules import define_rules_and_variables


def build_fuzzy_simulator():
    """ساخت شبیه‌ساز سیستم فازی بر اساس قوانین فایل fuzzy_rules"""
    rules = define_rules_and_variables()
    purchase_control = ctrl.ControlSystem(rules)
    purchase_simulator = ctrl.ControlSystemSimulation(purchase_control)
    return purchase_simulator


def evaluate_deal_value(price_diff_percent, body_score):
    """
    محاسبه مقدار نهایی غیرفازی‌شده ارزش خرید خودرو
    price_diff_percent: تفاوت قیمت به درصد (مثلا 10 یا 15-)
    body_score: امتیاز بدنه بین 0 تا 100
    """
    try:
        # ۱. ساخت شبیه‌ساز
        simulator = build_fuzzy_simulator()

        # ۲. محدود کردن ورودی‌ها به بازه مجاز جهت جلوگیری از ارورهای احتمالی
        price_diff_percent = max(min(price_diff_percent, 50), -50)
        body_score = max(min(body_score, 100), 0)

        # ۳. اعمال ورودی‌ها به سیستم فازی
        simulator.input['price_diff'] = price_diff_percent
        simulator.input['body_condition'] = body_score

        # ۴. اجرای فرآیند استنتاج و فازی‌زدایی مرکز ثقل
        simulator.compute()

        # ۵. بازگرداندن خروجی نهایی عددی
        result_value = simulator.output['purchase_value']
        return round(result_value, 2)

    except Exception as e:
        print(f"خطا در شبیه‌سازی سیستم فازی: {str(e)}")
        return None


# نمونه تستی برای تست صحت ارتباط دو فایل:
if __name__ == "__main__":
    # تست خودرویی که ۱۰٪ ارزان‌تر از کارشناسی است و بدنه عالی (۸۵) دارد
    test_price_diff = -10
    test_body_score = 85

    deal_value = evaluate_deal_value(test_price_diff, test_body_score)
    print(f"تفاوت قیمت: {test_price_diff}% | نمره بدنه: {test_body_score}/100")
    print(f"ارزش خرید پیشنهادی سیستم فازی: {deal_value} از ۱۰۰")