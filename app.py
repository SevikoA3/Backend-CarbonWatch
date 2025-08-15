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
from urllib.parse import urlparse

# Lazy imports for heavy libs
try:
    import tensorflow as tf
except Exception:
    tf = None

# Optional GCS client (used when MODEL_GCS_URI etc. are provided)
try:
    from google.cloud import storage
except Exception:
    storage = None

app = Flask(__name__)

# Allow overriding paths via environment (useful for Cloud Run)
MODEL_PATH = os.environ.get('MODEL_PATH', './carbon_watch_model.keras')
PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', './preprocessor.pkl')
LABEL_ENCODER_PATH = os.environ.get('LABEL_ENCODER_PATH', './label_encoder.pkl')

# Optionally provide GS URIs for artifacts. Examples:
# MODEL_GCS_URI=gs://bucket/path/carbon_watch_model.keras
# or set ARTIFACTS_GCS_PREFIX=gs://bucket/path to fetch standard filenames from that prefix
MODEL_GCS_URI = os.environ.get('MODEL_GCS_URI')
PREPROCESSOR_GCS_URI = os.environ.get('PREPROCESSOR_GCS_URI')
LABEL_ENCODER_GCS_URI = os.environ.get('LABEL_ENCODER_GCS_URI')
ARTIFACTS_GCS_PREFIX = os.environ.get('ARTIFACTS_GCS_PREFIX')

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

    # If GCS URIs or prefix are provided, download artifacts into the local paths.
    def _download_gcs_uri(uri, dest_path):
        if storage is None:
            app.logger.error('google-cloud-storage is required to download artifacts from GCS but is not installed')
            raise RuntimeError('google-cloud-storage not available')
        parsed = urlparse(uri)
        if parsed.scheme != 'gs':
            raise ValueError('GCS URI must start with gs://')
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip('/')
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        app.logger.info('Downloading %s from GCS bucket %s to %s', blob_name, bucket_name, dest_path)
        blob.download_to_filename(dest_path)

    # Resolve URIs from prefix if necessary
    if ARTIFACTS_GCS_PREFIX:
        prefix = ARTIFACTS_GCS_PREFIX.rstrip('/')
        if not prefix.startswith('gs://'):
            app.logger.error('ARTIFACTS_GCS_PREFIX must be a gs:// URI')
            raise ValueError('ARTIFACTS_GCS_PREFIX must be a gs:// URI')
        MODEL_GCS_URI = os.environ.get('MODEL_GCS_URI') or f"{prefix}/carbon_watch_model.keras"
        PREPROCESSOR_GCS_URI = os.environ.get('PREPROCESSOR_GCS_URI') or f"{prefix}/preprocessor.pkl"
        LABEL_ENCODER_GCS_URI = os.environ.get('LABEL_ENCODER_GCS_URI') or f"{prefix}/label_encoder.pkl"

    # Download any artifacts provided via GCS URIs
    for uri, local in [(MODEL_GCS_URI, MODEL_PATH), (PREPROCESSOR_GCS_URI, PREPROCESSOR_PATH), (LABEL_ENCODER_GCS_URI, LABEL_ENCODER_PATH)]:
        if uri:
            try:
                _download_gcs_uri(uri, local)
            except Exception as e:
                app.logger.exception('Failed to download %s -> %s: %s', uri, local, e)
                raise

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
