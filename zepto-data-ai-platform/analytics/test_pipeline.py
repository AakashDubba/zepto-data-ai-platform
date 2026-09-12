"""
Zepto Analytics — Pipeline Reload Test
=======================================
Reloads best_pipeline.joblib and validates predictions on raw, unpreprocessed input.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ANALYTICS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_pipeline():
    """Reload saved pipeline and validate predictions on raw data."""
    print("=" * 60)
    print("  Pipeline Reload Test")
    print("=" * 60)
    
    # 1. Load pipeline
    joblib_path = os.path.join(ANALYTICS_DIR, "best_pipeline.joblib")
    assert os.path.exists(joblib_path), f"Pipeline not found at {joblib_path}"
    
    pipeline = joblib.load(joblib_path)
    print(f"\n  ✅ Pipeline loaded from: {joblib_path}")
    print(f"     Pipeline steps: {[step[0] for step in pipeline.steps]}")
    
    # 2. Create raw, unpreprocessed test data (simulating new unseen data)
    raw_data = pd.DataFrame({
        'pclass': [1, 3, 2, 1, 3],
        'sex': ['female', 'male', 'female', 'male', 'female'],
        'age': [29.0, 22.0, np.nan, 45.0, 8.0],        # includes NaN to test imputation
        'sibsp': [0, 1, 0, 1, 3],
        'parch': [0, 0, 0, 0, 1],
        'fare': [211.3, 7.25, 13.0, np.nan, 21.075],    # includes NaN to test imputation
        'embarked': ['S', 'S', 'C', 'S', 'S'],
        'who': ['woman', 'man', 'woman', 'man', 'child'],
        'alone': [True, False, True, False, False],
    })
    
    print(f"\n  Raw input data (unpreprocessed, with NaN values):")
    print(raw_data.to_string(index=True))
    
    # 3. Predict
    predictions = pipeline.predict(raw_data)
    probabilities = pipeline.predict_proba(raw_data)
    
    print(f"\n  Predictions: {predictions.tolist()}")
    print(f"  Survival probabilities:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "Survived" if pred == 1 else "Not Survived"
        print(f"    Passenger {i+1}: {label} (P(survive) = {prob[1]:.4f})")
    
    # 4. Validate outputs
    assert len(predictions) == len(raw_data), "Prediction count mismatch"
    assert all(p in [0, 1] for p in predictions), "Invalid prediction values"
    assert probabilities.shape == (len(raw_data), 2), "Invalid probability shape"
    assert all(abs(p.sum() - 1.0) < 1e-6 for p in probabilities), "Probabilities don't sum to 1"
    
    print(f"\n  ✅ All validations passed!")
    print(f"     - {len(predictions)} predictions generated from raw data")
    print(f"     - Pipeline handled NaN values (imputation worked)")
    print(f"     - All predictions are valid binary values (0 or 1)")
    print(f"     - All probabilities sum to 1.0")
    
    print("\n" + "=" * 60)
    print("  PIPELINE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
