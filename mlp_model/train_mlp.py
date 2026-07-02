# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib


def train_and_save_model(csv_path):
    try:
        # ۱. خواندن فایل CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"فایل داده در مسیر '{csv_path}' یافت نشد.")

        df = pd.read_csv(csv_path)

        # ۲. جدا کردن ویژگی‌های ورودی (X) و هدف (y)
        required_columns = ['year', 'mileage', 'condition_score', 'price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ستون حیاتی '{col}' در فایل CSV یافت نشد.")

        X = df[['year', 'mileage', 'condition_score']].values
        y = df['price'].values.reshape(-1, 1)  # تبدیل به آرایه دوبعدی برای اسکیلر

        # ۳. تعریف و فیت کردن اسکیلرها برای نرمال‌سازی
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # ۴. تقسیم داده‌ها به آموزش و تست (۸۰ به ۲۰)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.20, random_state=42
        )

        # ۵. تعریف و آموزش مدل MLPRegressor
        # تعریف دو لایه پنهان با ساختار (32, 16)
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )

        print("در حال آموزش مدل MLPRegressor...")
        mlp.fit(X_train, y_train.ravel())  # مسطح کردن y برای برازش راحت‌تر مدل

        # ارزیابی اولیه مدل روی داده‌های تست
        score = mlp.score(X_test, y_test.ravel())
        print(f"مدل با موفقیت آموزش دید. ضریب تعیین (R2 Score) روی داده‌های تست: {score:.4f}")

        # ۶. ذخیره کردن مدل و اسکیلرها در پوشه saved_models
        output_dir = 'saved_models'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        joblib.dump(mlp, os.path.join(output_dir, 'model.pkl'))
        joblib.dump(scaler_x, os.path.join(output_dir, 'scaler_x.pkl'))
        joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))

        print(f"مدل و اسکیلرها با موفقیت در پوشه '{output_dir}' ذخیره شدند.")

    except Exception as e:
        print(f"خطا در فرآیند آموزش و ذخیره‌سازی: {str(e)}")

# برای اجرای نمونه (در صورت وجود فایل دیتاست):
# train_and_save_model('car_data.csv')