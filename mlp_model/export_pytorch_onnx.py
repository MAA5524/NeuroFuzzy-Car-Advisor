import torch
import joblib
import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from pytorch_model import CarPriceMLP

def export_model():
    model_dir = os.path.join(script_dir, 'saved_models')
    onnx_path = os.path.join(model_dir, 'pytorch_mlp.onnx')
    pkl_path = os.path.join(model_dir, 'pytorch_preprocessor.pkl')
    pth_path = os.path.join(model_dir, 'pytorch_model.pth')
    
    if not os.path.exists(pkl_path) or not os.path.exists(pth_path):
        print("Model files not found! Please run train_pytorch_mlp.ipynb first.")
        return
        
    print("Loading test data sample...")
    test_data = pd.read_csv(os.path.join(script_dir, '..', 'data', 'test_clean.csv'))
    X_sample = test_data.drop(columns=['price']).head(1)
    
    print("Loading Preprocessor...")
    preprocessor = joblib.load(pkl_path)
    
    # Preprocess the sample to get the correct input dimension for ONNX
    X_sample_processed = preprocessor.transform(X_sample)
    input_dim = X_sample_processed.shape[1]
    
    print("Loading PyTorch Model...")
    model = CarPriceMLP(input_dim)
    model.load_state_dict(torch.load(pth_path, weights_only=True))
    model.eval()
    
    print("Exporting PyTorch model to ONNX...")
    dummy_input = torch.tensor(X_sample_processed, dtype=torch.float32)
    
    torch.onnx.export(
        model, 
        (dummy_input,), 
        onnx_path, 
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model successfully saved as ONNX at {onnx_path}")
    print("NOTE: The ONNX file only contains the Neural Network. We will use the sklearn preprocessor (.pkl) before passing data to ONNX.")

if __name__ == '__main__':
    export_model()
