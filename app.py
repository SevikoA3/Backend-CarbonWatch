import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import gc
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

def get_supabase_client():
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_artifacts_to_cloud():
    """Upload trained model artifacts to Google Cloud Storage"""
    if storage is None:
        raise RuntimeError('google-cloud-storage is required to upload artifacts to GCS but is not installed')
    
    print("\n--- Uploading artifacts to Google Cloud Storage ---")
    
    # Define upload tasks
    upload_tasks = []
    
    # Determine URIs for upload
    if MODEL_GCS_URI:
        upload_tasks.append((MODEL_PATH, MODEL_GCS_URI))
    elif ARTIFACTS_GCS_PREFIX:
        model_uri = f"{ARTIFACTS_GCS_PREFIX.rstrip('/')}/carbon_watch_model.keras"
        upload_tasks.append((MODEL_PATH, model_uri))
    
    if PREPROCESSOR_GCS_URI:
        upload_tasks.append((PREPROCESSOR_PATH, PREPROCESSOR_GCS_URI))
    elif ARTIFACTS_GCS_PREFIX:
        preprocessor_uri = f"{ARTIFACTS_GCS_PREFIX.rstrip('/')}/preprocessor.pkl"
        upload_tasks.append((PREPROCESSOR_PATH, preprocessor_uri))
    
    if LABEL_ENCODER_GCS_URI:
        upload_tasks.append((LABEL_ENCODER_PATH, LABEL_ENCODER_GCS_URI))
    elif ARTIFACTS_GCS_PREFIX:
        label_encoder_uri = f"{ARTIFACTS_GCS_PREFIX.rstrip('/')}/label_encoder.pkl"
        upload_tasks.append((LABEL_ENCODER_PATH, label_encoder_uri))
    
    if not upload_tasks:
        raise ValueError("No GCS URIs configured for upload. Set MODEL_GCS_URI, PREPROCESSOR_GCS_URI, LABEL_ENCODER_GCS_URI, or ARTIFACTS_GCS_PREFIX")
    
    # Upload each file
    client = storage.Client()
    uploaded_paths = {}
    
    for local_path, gcs_uri in upload_tasks:
        try:
            # Parse GCS URI
            parsed = urlparse(gcs_uri)
            if parsed.scheme != 'gs':
                raise ValueError(f'GCS URI must start with gs://, got: {gcs_uri}')
            
            bucket_name = parsed.netloc
            blob_name = parsed.path.lstrip('/')
            
            # Upload file
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            print(f"Uploading {local_path} -> {gcs_uri}")
            blob.upload_from_filename(local_path)
            
            # Store successful upload
            uploaded_paths[local_path] = gcs_uri
            print(f"✅ Successfully uploaded {local_path}")
            
        except Exception as e:
            print(f"❌ Failed to upload {local_path} to {gcs_uri}: {e}")
            raise
    
    print("--- Cloud upload completed ---")
    return uploaded_paths

def cleanup_local_artifacts():
    """Clean up local artifact files after successful cloud upload"""
    files_to_clean = [MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH]
    
    for file_path in files_to_clean:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Cleaned up local file: {file_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove {file_path}: {e}")

def load_and_clean_data_from_supabase():
    """Load dataset from Supabase transaction table and clean the data"""
    print("\n--- Loading data from Supabase ---")
    
    try:
        supabase = get_supabase_client()
        
        # Fetch all transactions from the table
        response = supabase.table('transaction').select('*').execute()
        
        if not response.data:
            raise ValueError("No data found in transaction table")
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        print(f"Loaded {len(df)} records from Supabase")
        
        # Check for required columns - adapt column names as needed
        required_columns = [
            'Transaction Amount', 'Carbon Volume', 'Price per Ton', 'Origin Country',
            'Cross-Border Flag', 'Buyer Industry', 'Sudden Transaction Spike',
            'Transaction Hour', 'Entity Type', 'Label'
        ]
        
        # Map possible column variations (adjust based on your actual Supabase table schema)
        column_mapping = {
            'transaction_amount': 'Transaction Amount',
            'carbon_volume': 'Carbon Volume', 
            'price_per_ton': 'Price per Ton',
            'origin_country': 'Origin Country',
            'cross_border_flag': 'Cross-Border Flag',
            'buyer_industry': 'Buyer Industry',
            'sudden_transaction_spike': 'Sudden Transaction Spike',
            'transaction_hour': 'Transaction Hour',
            'entity_type': 'Entity Type',
            'label': 'Label'
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Check if we have all required columns after mapping
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle inconsistencies (e.g., 'PT.' to 'PT')
        if 'Entity Type' in df.columns:
            df['Entity Type'] = df['Entity Type'].astype(str).str.replace('.', '', regex=False)
        
        # Imputation for numeric missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
            print("Numeric missing values filled with median")
        
        # Imputation for categorical missing values  
        categorical_cols = df.select_dtypes(include='object').columns.drop('Label', errors='ignore')
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
            print("Categorical missing values filled with mode")
        
        print("--- Data cleaning completed ---")
        return df
        
    except Exception as e:
        print(f"Error loading data from Supabase: {e}")
        raise

def preprocess_training_data(df):
    """Preprocess data for training"""
    print("\n--- Starting data preprocessing ---")
    
    # Select features and target
    feature_cols = [
        'Transaction Amount', 'Carbon Volume', 'Price per Ton', 'Origin Country',
        'Cross-Border Flag', 'Buyer Industry', 'Sudden Transaction Spike',
        'Transaction Hour', 'Entity Type'
    ]
    
    # Check if all feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")
        
    X = df[feature_cols]
    y = df['Label']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Convert to categorical for TensorFlow
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    # Setup preprocessing transformers
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )
    
    print("Numeric features will be scaled, categorical features will be encoded")
    print("--- Preprocessing completed ---")
    
    return X, y_categorical, preprocessor, label_encoder

def build_training_model(input_shape):
    """Build neural network model for training"""
    print("\n--- Building neural network model ---")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: Verified, Caution, High-Risk
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture created and compiled")
    return model

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


def generate_gemini_explanation(prediction_data, reasons, feature_importance=None):
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
Tingkat Keyakinan: {confidence:.2%}

Alasan teknis yang ditemukan:
"""
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
        
    except Exception as e:
        app.logger.warning(f"Gemini explanation failed: {e}")
        # Fallback to rule-based explanation
        return f"Sistem mendeteksi transaksi dengan tingkat keyakinan {confidence:.1%}. " + ". ".join(reasons[:2]) + "."


@app.route('/train', methods=['POST'])
def train_model():
    """Train the CarbonWatch model using data from Supabase"""
    global tf
    
    try:
        # Import TensorFlow if not already loaded
        if tf is None:
            try:
                import importlib
                tf = importlib.import_module('tensorflow')
                # Set TensorFlow log level
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            except Exception as e:
                return jsonify({'error': 'TensorFlow not available', 'details': str(e)}), 500
        
        # Load and clean data from Supabase
        try:
            cleaned_df = load_and_clean_data_from_supabase()
        except Exception as e:
            return jsonify({'error': 'Failed to load data from Supabase', 'details': str(e)}), 500
        
        print(f"Dataset loaded with shape: {cleaned_df.shape}")
        print(f"Label distribution:\n{cleaned_df['Label'].value_counts()}")
        
        # Preprocess data
        try:
            X, y, preprocessor, label_encoder = preprocess_training_data(cleaned_df)
        except Exception as e:
            return jsonify({'error': 'Data preprocessing failed', 'details': str(e)}), 500
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set - Features: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Testing set - Features: {X_test.shape}, Labels: {y_test.shape}")
        
        # Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        print(f"Processed training features shape: {X_train_processed.shape}")
        input_dim = X_train_processed.shape[1]
        
        # Build model
        model = build_training_model(input_dim)
        
        # Setup early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train model
        print("\n--- Starting model training ---")
        history = model.fit(
            X_train_processed, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        print("--- Training completed ---")
        
        # Evaluate model
        print("\n--- Evaluating model ---")
        loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
        
        # Make predictions for detailed evaluation
        predictions = model.predict(X_test_processed)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate per-class accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=label_encoder.classes_, 
            output_dict=True
        )
        
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Save model and preprocessors
        try:
            # Save locally first
            model.save(MODEL_PATH)
            
            with open(PREPROCESSOR_PATH, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            with open(LABEL_ENCODER_PATH, 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f"Model saved locally to {MODEL_PATH}")
            print(f"Preprocessor saved locally to {PREPROCESSOR_PATH}")
            print(f"Label encoder saved locally to {LABEL_ENCODER_PATH}")
            
            # Upload to cloud storage if not in local mode
            cloud_paths = {}
            if not LOCAL_MODE:
                try:
                    cloud_paths = upload_artifacts_to_cloud()
                    print("✅ Artifacts successfully uploaded to cloud storage")
                    
                    # Optional: Clean up local files after successful cloud upload
                    # Uncomment the line below if you want to remove local files after upload
                    # cleanup_local_artifacts()
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to upload to cloud storage: {e}")
                    # Continue execution - local files are still available
            
            # Update global artifacts to use the new model
            artifacts['model'] = model
            artifacts['preprocessor'] = preprocessor
            artifacts['label_encoder'] = label_encoder
            artifacts['loaded'] = True
            
        except Exception as e:
            return jsonify({'error': 'Failed to save model artifacts', 'details': str(e)}), 500
        
        # Prepare training summary
        training_epochs = len(history.history['accuracy'])
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Check for overfitting
        acc_diff = abs(final_train_acc - final_val_acc)
        loss_diff = abs(final_val_loss - final_train_loss)
        overfitting_detected = acc_diff > 0.05 or loss_diff > 0.1
        
        response_data = {
            'status': 'success',
            'message': 'Model training completed successfully',
            'training_info': {
                'total_samples': len(cleaned_df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': list(X.columns),
                'target_classes': label_encoder.classes_.tolist(),
                'training_epochs': training_epochs,
                'input_dimensions': input_dim
            },
            'performance_metrics': {
                'test_accuracy': float(accuracy),
                'test_loss': float(loss),
                'training_accuracy': float(final_train_acc),
                'validation_accuracy': float(final_val_acc),
                'training_loss': float(final_train_loss),
                'validation_loss': float(final_val_loss),
                'overfitting_detected': overfitting_detected
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'label_distribution': cleaned_df['Label'].value_counts().to_dict(),
            'model_artifacts': {
                'local_paths': {
                    'model_path': MODEL_PATH,
                    'preprocessor_path': PREPROCESSOR_PATH,
                    'label_encoder_path': LABEL_ENCODER_PATH
                },
                'cloud_paths': cloud_paths if not LOCAL_MODE else {},
                'storage_mode': 'local' if LOCAL_MODE else 'cloud'
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.exception('Training error: %s', e)
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500


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
            # Generate basic explanations for this prediction
            reasons = analyze_prediction_reasons(
                df.iloc[[i]], 
                preds[i], 
                feature_names
            )
            
            # Advanced SHAP explanation if available
            feature_importance = {}
            if shap is not None:
                try:
                    # Create explainer for single prediction
                    X_single = X_processed[i:i+1]
                    explainer = shap.Explainer(lambda x: artifacts['model'].predict(x), X_single)
                    shap_values = explainer(X_single)
                    
                    # Get feature importance
                    if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 1:
                        importance_values = shap_values.values[0][:, pred_indices[i]] if len(shap_values.values.shape) == 3 else shap_values.values[0]
                        
                        for j, feat_name in enumerate(feature_names):
                            if j < len(importance_values):
                                feature_importance[feat_name] = float(importance_values[j])
                    
                except Exception as e:
                    app.logger.warning(f"SHAP analysis failed for record {i}: {e}")
                finally:
                    # Clean up SHAP objects
                    try:
                        del shap_values, explainer
                        gc.collect()
                    except Exception:
                        pass
            
            # Prepare prediction data for Gemini
            prediction_data = {
                'label': str(labels[i]),
                'confidence': float(confidences[i]),
                'transaction_id': records[i].get('id', f'TRP{20250000 + i + 1}')
            }
            
            # Generate AI explanation using Gemini
            ai_explanation = generate_gemini_explanation(prediction_data, reasons, feature_importance)
            
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
