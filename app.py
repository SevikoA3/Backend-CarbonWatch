"""Flask backend for CarbonWatch model predictions
Endpoint:
  POST /predict  -> accepts JSON body with a single record or list of records (feature names must match training CSV columns excluding 'Label')

Example request bodies:
  {"feature1": 1.2, "feature2": "A", ...}
  {"instances": [{...}, {...}]}

Response:
  {"predictions": [{"label": "Verified", "confidence": 0.93, "probabilities": [0.93, 0.05, 0.02]}, ...]}

Notes:
  - Model and preprocessing artifacts are loaded from ../ML relative to this file.
  - Make sure the files `carbon_watch_model.keras`, `preprocessor.pkl`, and `label_encoder.pkl` exist in the ML/ folder.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Lazy imports for heavy libs
try:
    import tensorflow as tf
except Exception:
    tf = None

app = Flask(__name__)

MODEL_PATH = './carbon_watch_model.keras'
PREPROCESSOR_PATH = './preprocessor.pkl'
LABEL_ENCODER_PATH = './label_encoder.pkl'

artifacts = {
    'model': None,
    'preprocessor': None,
    'label_encoder': None,
    'loaded': False,
}

def load_artifacts():
    """Load model and preprocessing artifacts into memory."""
    global artifacts, tf
    if artifacts['loaded']:
        return

    # Check files exist
    missing = []
    for p in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        app.logger.error('Missing artifact files: %s', missing)
        return

    # Load model. Import tensorflow into the module-level `tf` variable if not already present.
    if tf is None:
        try:
            import importlib
            tf = importlib.import_module('tensorflow')
        except Exception as e:
            app.logger.error('TensorFlow import failed: %s', e)
            raise RuntimeError('TensorFlow is not available in the environment') from e

    try:
        artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        app.logger.error('Failed to load model: %s', e)
        raise

    # Load preprocessor and label encoder
    try:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            artifacts['preprocessor'] = pickle.load(f)
    except Exception as e:
        app.logger.error('Failed to load preprocessor: %s', e)
        raise

    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            artifacts['label_encoder'] = pickle.load(f)
    except Exception as e:
        app.logger.error('Failed to load label encoder: %s', e)
        raise

    artifacts['loaded'] = True
    app.logger.info('Artifacts loaded successfully.')


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': artifacts['loaded']
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint. Accepts JSON with a single record, a list of records, or a dict with key 'instances'.
    Feature names must match the training data columns (excluding 'Label')."""
    try:
        if not artifacts['loaded']:
            load_artifacts()
        if not artifacts['loaded']:
            return jsonify({'error': 'Model artifacts are missing on server. Check logs.'}), 500

        data = request.get_json(force=True)

        # Normalize input into list of records
        if isinstance(data, dict) and 'instances' in data:
            records = data['instances']
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            return jsonify({'error': 'Unrecognized JSON format. Send an object, a list, or {"instances": [...] }'}), 400

        if len(records) == 0:
            return jsonify({'error': 'Empty input'}), 400

        # Build DataFrame
        df = pd.DataFrame.from_records(records)

        # Transform features
        preprocessor = artifacts['preprocessor']
        X_processed = preprocessor.transform(df)

        # Predict
        model = artifacts['model']
        preds = model.predict(X_processed)

        pred_indices = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)

        # Decode labels
        le = artifacts['label_encoder']
        try:
            labels = le.inverse_transform(pred_indices)
        except Exception:
            # If label encoder isn't sklearn's LabelEncoder, try simple mapping
            labels = [str(int(i)) for i in pred_indices]

        # Build response
        results = []
        for i in range(len(records)):
            results.append({
                'label': str(labels[i]),
                'confidence': float(confidences[i]),
                'probabilities': [float(x) for x in preds[i].tolist()]
            })

        return jsonify({'predictions': results})

    except Exception as e:
        app.logger.exception('Prediction error: %s', e)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


if __name__ == '__main__':
    # For local testing only; in production use a WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
