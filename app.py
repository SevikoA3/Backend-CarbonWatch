import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import gc

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

# Optional Gemini for natural language explanations
try:
    from google import genai
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        gemini_client = None
except Exception:
    gemini_client = None

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

    if LOCAL_MODE:
        # LOCAL MODE: Use local files
        missing = []
        for p in [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            app.logger.error('Missing local artifact files: %s', missing)
            return
            
    else:
        # CLOUD MODE: Download from GCS if URIs are provided
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
                if not ARTIFACTS_GCS_PREFIX:  # Only warn if no prefix is set
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


def analyze_prediction_reasons(df_record, prediction_probs, feature_names, feature_importance=None):
    """Analyze prediction and generate reasons/explanations using both SHAP and rule-based analysis"""
    reasons = []
    warnings = []
    used_warnings = set()  # Track used warning types to prevent duplicates
    
    # Get the predicted class index
    pred_class = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs)
    
    # Convert single record to series for easier access
    if isinstance(df_record, pd.DataFrame):
        record = df_record.iloc[0] if len(df_record) > 0 else df_record
    else:
        record = df_record
    
    # SHAP-based analysis (if available and significant)
    shap_reasons_added = 0
    if feature_importance:
        # Sort features by absolute importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Lower threshold for SHAP since our values are small
        significance_threshold = 0.001  # Reduced from 0.1
        
        for feat_name, importance in sorted_features[:5]:  # Check top 5 features
            if abs(importance) > significance_threshold and shap_reasons_added < 2:
                direction = "meningkatkan" if importance > 0 else "menurunkan"
                if feat_name in record.index:
                    feat_value = record[feat_name]
                    reasons.append(f"Fitur '{feat_name}' (nilai: {feat_value}) {direction} probabilitas prediksi secara signifikan")
                    shap_reasons_added += 1
    
    # Rule-based analysis (backup/additional context)
    # Volume/Amount analysis - prioritize the largest amount to avoid duplicates
    amount_features = [col for col in feature_names if 'amount' in col.lower() or 'volume' in col.lower() or 'value' in col.lower()]
    max_amount = 0
    max_amount_feature = None
    max_amount_value = 0
    
    for feat in amount_features:
        if feat in record.index:
            value = record[feat]
            if isinstance(value, (int, float)) and abs(value) > max_amount:
                max_amount = abs(value)
                max_amount_feature = feat
                max_amount_value = value
    
    # Add warning only for the highest amount feature
    if max_amount_feature and 'volume_warning' not in used_warnings:
        if max_amount_value > 100000:  # Very high threshold
            warnings.append(f"Volume transaksi sangat tinggi dan tidak wajar (${max_amount_value:,.2f})")
            used_warnings.add('volume_warning')
        elif max_amount_value > 10000:  # High threshold
            warnings.append(f"Volume transaksi cukup tinggi untuk profil ini (${max_amount_value:,.2f})")
            used_warnings.add('volume_warning')
        elif max_amount_value < 100:  # Low threshold
            warnings.append(f"Volume transaksi sangat rendah (${max_amount_value:,.2f})")
            used_warnings.add('volume_warning')
    
    # Time-based analysis  
    time_features = [col for col in feature_names if 'time' in col.lower() or 'hour' in col.lower() or 'day' in col.lower()]
    for feat in time_features:
        if feat in record.index and 'time_warning' not in used_warnings:
            value = record[feat]
            if isinstance(value, (int, float)):
                if value < 6 or value > 22:  # Outside business hours
                    warnings.append("Melakukan transaksi di luar jam kerja normal")
                    used_warnings.add('time_warning')
                    break  # Only add one time warning
    
    # Cross-border and entity analysis
    cross_border_features = [col for col in feature_names if 'cross' in col.lower() or 'border' in col.lower()]
    for feat in cross_border_features:
        if feat in record.index and 'cross_border_warning' not in used_warnings:
            value = record[feat]
            if isinstance(value, (int, float)) and value > 0:
                warnings.append("Transaksi lintas negara dengan risiko tambahan")
                used_warnings.add('cross_border_warning')
                break
    
    # Entity type analysis
    entity_features = [col for col in feature_names if 'entity' in col.lower() or 'type' in col.lower()]
    for feat in entity_features:
        if feat in record.index and 'entity_warning' not in used_warnings:
            value = str(record[feat]).lower()
            if 'individual' in value or 'unverified' in value:
                warnings.append("Tipe entitas memiliki profil risiko tinggi")
                used_warnings.add('entity_warning')
                break
    
    # Frequency analysis
    freq_features = [col for col in feature_names if 'freq' in col.lower() or 'count' in col.lower() or 'spike' in col.lower()]
    for feat in freq_features:
        if feat in record.index and 'frequency_warning' not in used_warnings:
            value = record[feat]
            if isinstance(value, (int, float)):
                if value > 20:  # High frequency
                    warnings.append("Pola transaksi dengan frekuensi tinggi terdeteksi")
                    used_warnings.add('frequency_warning')
                    break
                elif 'spike' in feat.lower() and value > 0:
                    warnings.append("Terdeteksi lonjakan mendadak dalam pola transaksi")
                    used_warnings.add('frequency_warning')
                    break
    
    # Generate main reason based on prediction confidence and class
    if confidence > 0.8:
        if pred_class == 1:  # High-Risk
            reasons.append("Mendeteksi pola transaksi yang tidak wajar")
        elif pred_class == 0:  # Verified
            reasons.append("Transaksi terverifikasi dengan pola normal")
        else:  # Caution
            reasons.append("Pola transaksi memerlukan review lebih lanjut")
    else:
        reasons.append("Model tidak yakin dengan prediksi - perlu analisis manual")
    
    # Add rule-based warnings only if we need more reasons (limit to 3 total)
    available_slots = 3 - len(reasons)
    if available_slots > 0 and warnings:
        # Take unique warnings up to available slots
        reasons.extend(warnings[:available_slots])
    
    # Ensure we always have at least 2 reasons
    if len(reasons) == 1:
        if confidence > 0.9:
            reasons.append("Tingkat keyakinan model sangat tinggi berdasarkan pola historis")
        elif confidence > 0.7:
            reasons.append("Tingkat keyakinan model tinggi berdasarkan pola historis")
        else:
            reasons.append("Perlu verifikasi tambahan karena pola tidak jelas")
    
    return reasons


def generate_gemini_explanation(prediction_data, reasons, feature_importance=None, input_data=None):
    """Generate natural language explanation using Gemini"""
    if not gemini_client:
        return "Penjelasan AI tidak tersedia - API key tidak ditemukan"
    
    try:
        # Prepare context for Gemini
        label = prediction_data['label']
        confidence = prediction_data['confidence']
        transaction_id = prediction_data.get('transaction_id', 'Unknown')
        
        # Build prompt for Gemini
        prompt = f"""
Anda adalah AI assistant untuk sistem deteksi fraud CarbonWatch. Berikan penjelasan dalam Bahasa Indonesia yang mudah dipahami tentang hasil prediksi berikut:

Transaksi ID: {transaction_id}
Hasil Prediksi: {label}
Tingkat Keyakinan: {confidence:.2%}"""

        # Add all input data if available
        if input_data:
            prompt += "\n\nData Input Transaksi:\n"
            for key, value in input_data.items():
                if key != 'id':  # Skip transaction ID as it's already shown
                    prompt += f"- {key}: {value}\n"

        prompt += "\nAlasan teknis yang ditemukan:\n"
        for i, reason in enumerate(reasons, 1):
            prompt += f"{i}. {reason}\n"
        
        if feature_importance:
            prompt += "\nFitur yang paling berpengaruh:\n"
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feat, importance in sorted_features:
                prompt += f"- {feat}: {importance:.3f}\n"
        
        prompt += """
Tugas Anda:
1. Jelaskan hasil prediksi dengan bahasa yang mudah dipahami
2. Berikan rekomendasi tindakan yang jelas
3. Sebutkan tingkat risiko dan alasannya
4. Gunakan maksimal 3-4 kalimat yang informatif
5. Jawaban tidak perlu menggunakan formatting seperti bold, italic, dan lain-lain
6. Tuliskan penjelasan tanpa menggunakan kata ganti orang pertama (mis. "kami", "saya")

Format jawaban dalam bahasa Indonesia yang profesional dan tidak bertele-tele tapi mudah dipahami.
"""
        
        # Call Gemini API
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return response.text.strip()
        
    except Exception:
        # Fallback to rule-based explanation
        return f"Sistem mendeteksi transaksi dengan tingkat keyakinan {confidence:.1%}. " + ". ".join(reasons[:2]) + "."


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
            # Advanced SHAP explanation if available (do this first)
            feature_importance = {}
            if shap is not None:
                try:
                    # Create explainer for single prediction
                    X_single = X_processed[i:i+1]
                    
                    # Handle sparse matrices - get shape safely
                    n_samples = X_processed.shape[0]
                    
                    # Use larger background sample for better baseline
                    background_size = min(50, n_samples)  # Increased from 10 to 50
                    X_background = X_processed[:background_size]
                    
                    # Convert sparse to dense if needed for SHAP
                    if hasattr(X_single, 'toarray'):
                        X_single_dense = X_single.toarray()
                        X_background_dense = X_background.toarray()
                    else:
                        X_single_dense = X_single
                        X_background_dense = X_background
                    
                    # Use Kernel explainer for more robust results with neural networks
                    def model_predict_wrapper(x):
                        return artifacts['model'].predict(x, verbose=0)
                    
                    # Create explainer with proper background
                    explainer = shap.KernelExplainer(model_predict_wrapper, X_background_dense)
                    shap_values = explainer.shap_values(X_single_dense, nsamples=100)
                    
                    # Handle different SHAP output formats for multi-class
                    if isinstance(shap_values, list):
                        # Multi-class: shap_values is list of arrays, one per class
                        class_shap_values = shap_values[pred_indices[i]]
                        if len(class_shap_values.shape) == 2:
                            importance_values = class_shap_values[0]
                        else:
                            importance_values = class_shap_values
                    else:
                        # For KernelExplainer with multi-class: shape is (n_samples, n_features, n_classes)
                        if len(shap_values.shape) == 3:
                            importance_values = shap_values[0, :, pred_indices[i]]
                        elif len(shap_values.shape) == 2:
                            importance_values = shap_values[0]
                        else:
                            importance_values = shap_values
                    
                    # Create feature importance dict
                    for j, feat_name in enumerate(feature_names):
                        if j < len(importance_values):
                            raw_val = importance_values[j]
                            if hasattr(raw_val, 'item'):
                                importance_val = float(raw_val.item())
                            else:
                                importance_val = float(raw_val)
                            feature_importance[feat_name] = importance_val
                    
                    # Fallback if SHAP values are too small
                    non_zero_features = {k: v for k, v in feature_importance.items() if abs(v) > 0.001}
                    if not non_zero_features:
                        try:
                            # Use prediction perturbation as fallback
                            baseline_input = np.mean(X_background_dense, axis=0, keepdims=True)
                            current_pred = artifacts['model'].predict(X_single_dense, verbose=0)
                            
                            perturbation_size = 0.1
                            for idx, feat_name in enumerate(feature_names[:min(len(feature_names), X_single_dense.shape[1])]):
                                perturbed_input = X_single_dense.copy()
                                if idx < X_single_dense.shape[1]:
                                    perturbed_input[0, idx] += perturbation_size
                                    perturbed_pred = artifacts['model'].predict(perturbed_input, verbose=0)
                                    importance_val = float(abs(perturbed_pred[0, pred_indices[i]] - current_pred[0, pred_indices[i]]))
                                    feature_importance[feat_name] = importance_val
                        except Exception:
                            pass
                    
                except Exception as e:
                    # Complete fallback: simple feature ranking based on input values
                    input_values = X_single_dense[0] if hasattr(X_single_dense, 'shape') else X_single_dense
                    for idx, feat_name in enumerate(feature_names):
                        if idx < len(input_values):
                            feature_importance[feat_name] = float(abs(input_values[idx]) * 0.1)
                finally:
                    # Clean up SHAP objects
                    try:
                        del shap_values, explainer
                        gc.collect()
                    except Exception:
                        pass
            else:
                pass  # SHAP not available
            
            # Generate explanations using both SHAP and rule-based analysis
            reasons = analyze_prediction_reasons(
                df.iloc[[i]], 
                preds[i], 
                feature_names,
                feature_importance  # Pass SHAP values
            )
            
            # Prepare prediction data for Gemini
            prediction_data = {
                'label': str(labels[i]),
                'confidence': float(confidences[i]),
                'transaction_id': records[i].get('id', f'TRP{20250000 + i + 1}')
            }
            
            # Generate AI explanation using Gemini
            ai_explanation = generate_gemini_explanation(
                prediction_data, 
                reasons, 
                feature_importance,
                input_data=records[i]  # Pass all input data
            )
            
            result = {
                'label': prediction_data['label'],
                'confidence': prediction_data['confidence'],
                'probabilities': [float(x) for x in preds[i].tolist()],
                'transaction_id': prediction_data['transaction_id'],
                'explanation': {
                    'ai_summary': ai_explanation,
                    'technical_reasons': reasons,
                    'feature_importance': feature_importance if feature_importance else {},
                }
            }
            results.append(result)

        return jsonify({'predictions': results})

    except Exception as e:
        app.logger.exception('Prediction error: %s', e)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


if __name__ == '__main__':
    # For local testing only; in production use a WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
