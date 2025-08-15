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

# Optional SHAP for advanced explanations
try:
    import shap
except Exception:
    shap = None

# Optional GCS client (used when MODEL_GCS_URI etc. are provided)
try:
    from google.cloud import storage
except Exception:
    storage = None

app = Flask(__name__)

# Environment configuration
LOCAL_MODE = os.environ.get('LOCAL', 'true').lower() == 'true'

# Allow overriding paths via environment (useful for Cloud Run)
MODEL_PATH = os.environ.get('MODEL_PATH', './carbon_watch_model.keras')
PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', './preprocessor.pkl')
LABEL_ENCODER_PATH = os.environ.get('LABEL_ENCODER_PATH', './label_encoder.pkl')

# Cloud storage URIs (only used when LOCAL=false)
# Examples:
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

    app.logger.info(f'Loading artifacts in {"LOCAL" if LOCAL_MODE else "CLOUD"} mode')

    if LOCAL_MODE:
        # LOCAL MODE: Use local files
        app.logger.info('Using local files for artifacts')
        
        # Check if local files exist
        missing = []
        for p in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            app.logger.error('Missing local artifact files: %s', missing)
            return
            
    else:
        # CLOUD MODE: Download from GCS if URIs are provided
        app.logger.info('Using cloud storage for artifacts')
        
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
        model_uri = MODEL_GCS_URI
        preprocessor_uri = PREPROCESSOR_GCS_URI
        label_encoder_uri = LABEL_ENCODER_GCS_URI
        
        if ARTIFACTS_GCS_PREFIX:
            prefix = ARTIFACTS_GCS_PREFIX.rstrip('/')
            if not prefix.startswith('gs://'):
                app.logger.error('ARTIFACTS_GCS_PREFIX must be a gs:// URI')
                raise ValueError('ARTIFACTS_GCS_PREFIX must be a gs:// URI')
            model_uri = model_uri or f"{prefix}/carbon_watch_model.keras"
            preprocessor_uri = preprocessor_uri or f"{prefix}/preprocessor.pkl"
            label_encoder_uri = label_encoder_uri or f"{prefix}/label_encoder.pkl"

        # Download artifacts from GCS
        download_tasks = [
            (model_uri, MODEL_PATH),
            (preprocessor_uri, PREPROCESSOR_PATH), 
            (label_encoder_uri, LABEL_ENCODER_PATH)
        ]
        
        for uri, local_path in download_tasks:
            if uri:
                try:
                    _download_gcs_uri(uri, local_path)
                except Exception as e:
                    app.logger.exception('Failed to download %s -> %s: %s', uri, local_path, e)
                    raise
            else:
                app.logger.warning('No URI provided for %s', local_path)
                
        # Check that files exist after download
        missing = []
        for p in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            app.logger.error('Missing artifact files after download: %s', missing)
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


def analyze_prediction_reasons(df_record, prediction_probs, feature_names):
    """Analyze prediction and generate reasons/explanations"""
    reasons = []
    warnings = []
    
    # Get the predicted class index
    pred_class = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs)
    
    # Convert single record to series for easier access
    if isinstance(df_record, pd.DataFrame):
        record = df_record.iloc[0] if len(df_record) > 0 else df_record
    else:
        record = df_record
    
    # Analyze different aspects based on common carbon/transaction features
    # You should customize these rules based on your actual features
    
    # Volume/Amount analysis
    amount_features = [col for col in feature_names if 'amount' in col.lower() or 'volume' in col.lower() or 'value' in col.lower()]
    for feat in amount_features:
        if feat in record.index:
            value = record[feat]
            if isinstance(value, (int, float)):
                if value > 10000:  # Adjust threshold based on your data
                    warnings.append(f"Volume transaksi tidak proporsional dengan profil industri (${value:,.2f})")
                elif value < 100:
                    warnings.append(f"Volume transaksi sangat rendah (${value:,.2f})")
    
    # Time-based analysis  
    time_features = [col for col in feature_names if 'time' in col.lower() or 'hour' in col.lower() or 'day' in col.lower()]
    for feat in time_features:
        if feat in record.index:
            value = record[feat]
            if isinstance(value, (int, float)):
                if value < 6 or value > 22:  # Outside business hours
                    warnings.append("Melakukan transaksi di luar jam kerja normal")
    
    # Frequency analysis
    freq_features = [col for col in feature_names if 'freq' in col.lower() or 'count' in col.lower()]
    for feat in freq_features:
        if feat in record.index:
            value = record[feat]
            if isinstance(value, (int, float)):
                if value > 20:  # High frequency
                    warnings.append("Pola transaksi dengan frekuensi tinggi terdeteksi")
                elif value < 2:
                    warnings.append("Aktivitas transaksi sangat jarang")
    
    # Category/Type analysis
    category_features = [col for col in feature_names if 'category' in col.lower() or 'type' in col.lower()]
    for feat in category_features:
        if feat in record.index:
            value = str(record[feat]).lower()
            if 'unusual' in value or 'irregular' in value:
                warnings.append("Kategori transaksi tidak umum terdeteksi")
    
    # Generate main reason based on prediction confidence and class
    if confidence > 0.8:
        if pred_class == 1:  # Adjust based on your label encoding
            reasons.append("Mendeteksi pola transaksi yang tidak wajar")
        elif pred_class == 0:
            reasons.append("Transaksi terverifikasi dengan pola normal")
        else:
            reasons.append("Pola transaksi memerlukan review lebih lanjut")
    else:
        reasons.append("Model tidak yakin dengan prediksi - perlu analisis manual")
    
    # Add warnings as reasons
    if warnings:
        reasons.extend(warnings[:3])  # Limit to top 3 warnings
    
    # If no specific reasons found, add generic ones based on confidence
    if len(reasons) == 1:
        if confidence > 0.7:
            reasons.append("Tingkat keyakinan model tinggi berdasarkan pola historis")
        else:
            reasons.append("Perlu verifikasi tambahan karena pola tidak jelas")
    
    return reasons


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': artifacts['loaded'],
        'mode': 'LOCAL' if LOCAL_MODE else 'CLOUD',
        'local_files_exist': all(os.path.exists(p) for p in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]) if LOCAL_MODE else 'N/A',
        'cloud_config': {
            'gcs_prefix': ARTIFACTS_GCS_PREFIX,
            'model_uri': MODEL_GCS_URI,
            'preprocessor_uri': PREPROCESSOR_GCS_URI,
            'label_encoder_uri': LABEL_ENCODER_GCS_URI
        } if not LOCAL_MODE else 'N/A'
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

        # Get feature names for analysis
        feature_names = df.columns.tolist()

        # Build response with explanations
        results = []
        for i in range(len(records)):
            # Generate explanations for this prediction
            reasons = analyze_prediction_reasons(
                df.iloc[[i]], 
                preds[i], 
                feature_names
            )
            
            result = {
                'label': str(labels[i]),
                'confidence': float(confidences[i]),
                'probabilities': [float(x) for x in preds[i].tolist()],
                'reasons': reasons,
                'transaction_id': records[i].get('id', f'TRP{20250000 + i + 1}'),  # Generate ID if not provided
                'analysis': {
                    'risk_level': 'HIGH' if confidences[i] > 0.8 and pred_indices[i] == 0 else 'MEDIUM' if confidences[i] < 0.6 else 'LOW',
                    'recommendation': 'Perlu investigasi lebih lanjut' if confidences[i] > 0.8 and pred_indices[i] == 0 else 'Transaksi dapat diproses'
                }
            }
            results.append(result)

        return jsonify({'predictions': results})

    except Exception as e:
        app.logger.exception('Prediction error: %s', e)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_prediction():
    """Enhanced explanation endpoint with SHAP analysis if available"""
    try:
        if not artifacts['loaded']:
            load_artifacts()
        if not artifacts['loaded']:
            return jsonify({'error': 'Model artifacts are missing on server. Check logs.'}), 500

        data = request.get_json(force=True)
        
        # Normalize input into single record
        if isinstance(data, dict) and 'instances' in data:
            record = data['instances'][0] if data['instances'] else {}
        elif isinstance(data, list):
            record = data[0] if data else {}
        elif isinstance(data, dict):
            record = data
        else:
            return jsonify({'error': 'Send a single record for detailed explanation'}), 400

        # Build DataFrame
        df = pd.DataFrame([record])
        
        # Transform features
        preprocessor = artifacts['preprocessor']
        X_processed = preprocessor.transform(df)
        
        # Predict
        model = artifacts['model']
        preds = model.predict(X_processed)
        pred_idx = np.argmax(preds[0])
        confidence = np.max(preds[0])
        
        # Decode label
        le = artifacts['label_encoder']
        try:
            label = le.inverse_transform([pred_idx])[0]
        except Exception:
            label = str(int(pred_idx))
        
        # Get basic reasons
        reasons = analyze_prediction_reasons(df, preds[0], df.columns.tolist())
        
        # Advanced SHAP explanation if available
        feature_importance = {}
        shap_explanation = None
        
        if shap is not None:
            try:
                # Create explainer (you might want to cache this)
                explainer = shap.Explainer(lambda x: artifacts['model'].predict(x), X_processed[:1])
                shap_values = explainer(X_processed[:1])
                
                # Get feature importance
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 1:
                    importance_values = shap_values.values[0][:, pred_idx] if len(shap_values.values.shape) == 3 else shap_values.values[0]
                    feature_names = df.columns.tolist()
                    
                    for i, feat_name in enumerate(feature_names):
                        if i < len(importance_values):
                            feature_importance[feat_name] = float(importance_values[i])
                
                shap_explanation = "SHAP analysis completed - features ranked by importance"
            except Exception as e:
                app.logger.warning(f"SHAP analysis failed: {e}")
                shap_explanation = "SHAP analysis unavailable"
        
        # Enhanced response
        response = {
            'prediction': {
                'label': str(label),
                'confidence': float(confidence),
                'probabilities': [float(x) for x in preds[0].tolist()],
                'transaction_id': record.get('id', f'TRP{20250001}')
            },
            'explanation': {
                'main_reasons': reasons,
                'feature_importance': feature_importance,
                'shap_analysis': shap_explanation,
                'risk_assessment': {
                    'level': 'HIGH' if confidence > 0.8 and pred_idx == 0 else 'MEDIUM' if confidence < 0.6 else 'LOW',
                    'score': float(confidence),
                    'recommendation': 'Perlu investigasi segera' if confidence > 0.9 and pred_idx == 0 
                                   else 'Monitoring diperlukan' if confidence > 0.7 and pred_idx == 0
                                   else 'Transaksi dapat diproses normal'
                }
            },
            'input_analysis': {
                'features_analyzed': len(df.columns),
                'suspicious_patterns': len([r for r in reasons if any(word in r.lower() for word in ['tidak', 'tinggi', 'rendah', 'luar'])]),
                'data_quality': 'Good' if not df.isnull().any().any() else 'Has missing values'
            }
        }
        
        return jsonify(response)

    except Exception as e:
        app.logger.exception('Explanation error: %s', e)
        return jsonify({'error': 'Explanation failed', 'details': str(e)}), 500


if __name__ == '__main__':
    # For local testing only; in production use a WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
