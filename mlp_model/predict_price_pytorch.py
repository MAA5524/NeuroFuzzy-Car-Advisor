import os
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt

def predict_car_price(features_dict):
    """
    Predicts the car price using the saved Scikit-Learn Preprocessor and PyTorch ONNX model.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    onnx_path = os.path.join(model_dir, 'pytorch_mlp.onnx')
    pkl_path = os.path.join(model_dir, 'pytorch_preprocessor.pkl')
    
    if not os.path.exists(onnx_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError("Model files (ONNX or PKL) not found. Train and export the model first.")
        
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # 1. Preprocess using Scikit-Learn Pipeline
    preprocessor = joblib.load(pkl_path)
    X_processed = preprocessor.transform(input_df)
    
    # 2. Predict using ONNX (PyTorch)
    sess = rt.InferenceSession(onnx_path)
    inputs = {sess.get_inputs()[0].name: X_processed.astype(np.float32)}
    pred_scaled = sess.run(None, inputs)[0]
    
    # 3. Inverse Target Transform
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    target_scaler = joblib.load(target_scaler_path)
    predicted_price = target_scaler.inverse_transform(pred_scaled)[0][0]
    
    return float(predicted_price)

if __name__ == '__main__':
    # Sample Test
    sample_car = {
        'year': 2018,
        'mileage': 65000,
        'insurance': 6,
        'car_age': 8,
        'yearly_mileage': 8125.0,
        'has_insurance': 1,
        'gearbox_type': 'اتوماتیک',
        'gearbox_health': 'سالم و پلمپ',
        'motor_status': 'سالم',
        'body_status': 'سالم و بی‌خط و خش',
        'chassis_status': 'سالم و پلمپ',
        'brand': 'peugeot_206'
    }
    
    try:
        price = predict_car_price(sample_car)
        print(f"Predicted Price: {price:,.0f} Tomans")
    except Exception as e:
        print(f"Error: {e}")
