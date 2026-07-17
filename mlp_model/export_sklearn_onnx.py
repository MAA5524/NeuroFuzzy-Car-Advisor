import joblib
import pandas as pd
import onnx
import sys
# Monkeypatch for skl2onnx compatibility with ONNX >= 1.16
if not hasattr(onnx, 'mapping') and hasattr(onnx, '_mapping'):
    _onnx_mapping = getattr(onnx, '_mapping')
    sys.modules['onnx.mapping'] = _onnx_mapping
    setattr(onnx, 'mapping', _onnx_mapping)

import onnx.helper
if not hasattr(onnx.helper, 'split_complex_to_pairs'):
    def split_complex_to_pairs(ca):
        return [(ca[i // 2].real if (i % 2 == 0) else ca[i // 2].imag) for i in range(len(ca) * 2)]
    setattr(onnx.helper, 'split_complex_to_pairs', split_complex_to_pairs)

from skl2onnx import to_onnx
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'saved_models')
pipeline_path = os.path.join(model_dir, 'sklearn_mlp.pkl')
onnx_path = os.path.join(model_dir, 'sklearn_mlp.onnx')

print("Loading Pipeline model...")
pipeline = joblib.load(pipeline_path)

# We need a raw data sample (Initial Type) so skl2onnx can infer input types
print("Loading test data sample...")
test_data = pd.read_csv(os.path.join(script_dir, '..', 'data', 'test_clean.csv'))
X_sample = test_data.drop(columns=['price']).head(1)

# Converting DataFrame data types to standard Python types if needed (usually skl2onnx handles this)
# pandas Float64 gets converted to Python Float
print("Converting to ONNX format...")
onx = to_onnx(pipeline, X_sample, target_opset=12)

# to_onnx returns (onnx_model, topology) if intermediate=True, otherwise just onnx_model.
# We check if it is a tuple and unpack it for static type safety.
if isinstance(onx, tuple):
    onx = onx[0]

with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())
    
print(f"Model successfully saved as ONNX at {onnx_path}")

