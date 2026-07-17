import os
import joblib
import json
import numpy as np

def export_json():
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    pkl_path = os.path.join(model_dir, 'pytorch_preprocessor.pkl')
    json_path = os.path.join(model_dir, 'preprocessing_config.json')
    
    if not os.path.exists(pkl_path):
        print("Preprocessor PKL not found! Please train the model first.")
        return
        
    preprocessor = joblib.load(pkl_path)
    
    # 1. Get StandardScaler parameters
    num_transformer = preprocessor.named_transformers_['num']
    mean_vals = num_transformer.mean_.tolist()
    scale_vals = num_transformer.scale_.tolist()
    
    # 2. Get OneHotEncoder categories
    cat_transformer = preprocessor.named_transformers_['cat']
    categories = [cat.tolist() for cat in cat_transformer.categories_]
    
    # 3. Get Target Scaler parameters
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    target_scaler = joblib.load(target_scaler_path)
    target_mean = target_scaler.mean_.tolist()
    target_scale = target_scaler.scale_.tolist()
    
    numeric_features = ['year', 'mileage', 'car_age', 'yearly_mileage', 'has_insurance']
    categorical_features = ['gearbox_type', 'gearbox_health', 'motor_status', 'body_status', 'chassis_status', 'brand']
    
    config = {
        "numeric_features": numeric_features,
        "standard_scaler": {
            "mean": mean_vals,
            "scale": scale_vals
        },
        "categorical_features": categorical_features,
        "one_hot_categories": categories,
        "target_scaler": {
            "mean": target_mean,
            "scale": target_scale
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
    print(f"Preprocessing config successfully saved to {json_path}")

if __name__ == '__main__':
    export_json()
