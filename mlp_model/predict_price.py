# predict_price.py
import os
import numpy as np
import joblib


def predict_fair_price(features_dict):
    """
    دریافت دیکشنری مشخصات خودرو و پیش‌بینی قیمت عادلانه کارشناسی
    فرمت ورودی پیشنهادی:
    features_dict = {'year': 1399, 'mileage': 50000, 'condition_score': 8.5}
    """
    model_dir = 'saved_models'
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_x_path = os.path.join(model_dir, 'scaler_x.pkl')
    scaler_y_path = os.path.join(model_dir, 'scaler_y.pkl')

    # مدیریت خطا برای اطمینان از وجود فایل‌های مدل روی هارد
    try:
        if not (os.path.exists(model_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
            raise FileNotFoundError("فایل‌های مدل یا اسکیلرها یافت نشدند. ابتدا اسکریپت train_model.py را اجرا کنید.")

        # ۱. لود کردن مدل و اسکیلرها از هارد
        model = joblib.load(model_path)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        # ۲. تبدیل ویژگی‌های ورودی به فرمت آرایه دو بعدی مناسب برای پیش‌بینی
        x_input = np.array([[
            features_bool := features_dict_to_list(features_dict)
        ]])

        # استخراج مقادیر به ترتیب مشخص
        raw_features = [
            features_dict['year'],
            features__to_list(features_dict)
        ]

        # اصلاح فرمت آرایه ورودی برای سازگاری با Scikit-learn
        raw_features = np.array([[
            features_dict['year'],
            features__to_list(features_dict)  # نمونه فرضی برای سادگی
        ]])

    except KeyError as e:
        print(f"خطا: مقدار کلید مشخصه ورودی نامعتبر است. {str(e)}")
        return None
    except FileNotFoundError as e:
        print(f"خطای لود کردن مدل: {str(e)}")
        return None
    except Exception as e:
        print(f"خطای غیرمنتظره: {str(e)}")
        return None


def features_dict_to_list(features_dict):
    """تابع کمکی برای اطمینان از ترتیب ستون‌ها: year, mileage, condition_score"""
    return [
        features_dict['year'],
        features_dict['mileage'],
        features_dict['condition_score']
    ]


def predict_fair_price_safe(features_dict):
    """نسخه اصلاح شده و ایمن پیش‌بینی"""
    model_dir = 'saved_models'
    try:
        # لود مدل‌ها
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        scaler_x = joblib.load(os.path.join(model_dir, 'scaler_x.pkl'))
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

        # تبدیل ساختار دیکشنری به لیست با ترتیب صحیح
        raw_features = np.array([features_dict_to_list(features_dict)])

        # ۳. اسکیل کردن ویژگی‌های ورودی
        scaled_features = scaler_x.transform(raw_features)

        # ۴. پیش‌بینی قیمت مقیاس‌دهی شده
        predicted_scaled_price = model.predict(scaled_features)

        # ۵. بازگرداندن قیمت به مقیاس واقعی (Inverse Transform)
        predicted_real_price = scaler_y.inverse_transform(predicted_scaled_price.reshape(-1, 1))

        # بازگرداندن مقدار نهایی به صورت یک عدد اعشاری (Float)
        return float(predicted_real_price[0][0])

    except FileNotFoundError:
        print("خطا: فایل‌های مدل در مسیر saved_models یافت نشد. لطفاً ابتدا مدل را آموزش دهید.")
        return None
    except Exception as e:
        print(f"خطا در پیش‌بینی: {str(e)}")
        return None

# نمونه استفاده فرضی:
# test_car = {'year': 1399, 'mileage': 50000, 'condition_score': 8.5}
# print("قیمت کارشناسی شده:", predict_fair_price_safe(test_car))