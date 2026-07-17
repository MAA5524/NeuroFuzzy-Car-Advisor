# predict_price.py
import os
import joblib
import pandas as pd
import numpy as np

try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def predict_fair_price(features_dict):
    """
    Predict fair price using ONNX pipeline or Pickle file.
    This function expects a dictionary containing raw string/numeric values, e.g.:
    {
        'year': 2020, 'mileage': 50000, 'car_age': 6, 'yearly_mileage': 8333.3,
        'has_insurance': 1, 'gearbox_type': 'اتوماتیک', 'gearbox_health': 'سالم و پلمپ',
        'motor_status': 'سالم', 'body_status': 'سالم و بی‌خط و خش',
        'chassis_status': 'سالم و پلمپ', 'brand': 'peugeot_206'
    }
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    onnx_path = os.path.join(model_dir, 'mlp_pipeline.onnx')
    pkl_path = os.path.join(model_dir, 'mlp_pipeline.pkl')
    
    # Convert dictionary to a DataFrame with a single row (standard format for our model)
    input_df = pd.DataFrame([features_dict])
    
    try:
        # 1. First, try to predict with ONNX (for much higher speed)
        if ONNX_AVAILABLE and os.path.exists(onnx_path):
            sess = rt.InferenceSession(onnx_path)
            
            # Convert DataFrame to dictionary of NumPy arrays for ONNX inputs
            inputs = {c: input_df[c].values for c in input_df.columns}
            
            # Fix data types (ONNX is highly sensitive to Types)
            for k, v in inputs.items():
                if input_df[k].dtype == 'int64':
                    inputs[k] = v.astype(np.int64)
                elif input_df[k].dtype == 'float64':
                    inputs[k] = v.astype(np.float64)
                elif input_df[k].dtype == 'object':
                    inputs[k] = v.astype(str)
            
            pred_onx = sess.run(None, inputs)
            predicted_scaled_price = float(pred_onx[0][0][0])
            
        # 2. If ONNX file doesn't exist, use the standard Scikit-Learn PKL file
        elif os.path.exists(pkl_path):
            pipeline = joblib.load(pkl_path)
            predicted_scaled_price = float(pipeline.predict(input_df)[0])
        else:
            raise FileNotFoundError("No model files (ONNX or PKL) found in the saved_models directory.")
            
        # 3. Inverse transform the scaled price using Target Scaler
        target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
        if not os.path.exists(target_scaler_path):
            raise FileNotFoundError("target_scaler.pkl not found for inverse transforming the price.")
            
        target_scaler = joblib.load(target_scaler_path)
        real_price = target_scaler.inverse_transform([[predicted_scaled_price]])[0][0]
        
        return float(real_price)
        
    except Exception as e:
        print(f"Error in predicting price: {str(e)}")
        return None

# # Quick test sample:
# if __name__ == '__main__':
#     test_features = {
#         'year': 2020, 'mileage': 50000, 'car_age': 6, 'yearly_mileage': 8333.3,
#         'has_insurance': 1, 'gearbox_type': 'اتوماتیک', 'gearbox_health': 'سالم و پلمپ',
#         'motor_status': 'سالم', 'body_status': 'سالم و بی‌خط و خش',
#         'chassis_status': 'سالم و پلمپ', 'brand': 'peugeot_206'
#     }
#     print("Predicted Price:", predict_fair_price(test_features))