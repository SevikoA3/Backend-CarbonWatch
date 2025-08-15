# CarbonWatch Backend

This Flask backend provides endpoints for carbon transaction fraud detection using a trained machine learning model.

## Endpoints

### 1. Prediction Endpoint (`/predict`)
Predicts fraud risk for carbon credit transactions.

### 2. Training Endpoint (`/train`)
Trains a new model using data from Supabase transaction table.

## Files Structure

Expected model artifacts:
- `carbon_watch_model.keras` - Trained Keras model
- `preprocessor.pkl` - Feature preprocessing pipeline
- `label_encoder.pkl` - Target label encoder

## Quick Start

1. **Environment Setup**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configure Environment Variables**:
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon/public key
- `GEMINI_API_KEY`: (Optional) For AI explanations

3. **Run the Application**:
```bash
python app.py
```

## API Usage

### Prediction Endpoint

**Single prediction**:
```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Transaction Amount": 1500000,
    "Carbon Volume": 30000,
    "Price per Ton": 25,
    "Origin Country": "Indonesia",
    "Cross-Border Flag": 1,
    "Buyer Industry": "Manufacturing",
    "Sudden Transaction Spike": 0,
    "Transaction Hour": 14,
    "Entity Type": "Corporation"
  }'
```

**Batch predictions**:
```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      {
        "Transaction Amount": 1500000,
        "Carbon Volume": 30000,
        "Price per Ton": 25,
        "Origin Country": "Indonesia",
        "Cross-Border Flag": 1,
        "Buyer Industry": "Manufacturing", 
        "Sudden Transaction Spike": 0,
        "Transaction Hour": 14,
        "Entity Type": "Corporation"
      }
    ]
  }'
```

### Training Endpoint

**Train new model**:
```bash
curl -X POST http://localhost:5000/train \
  -H 'Content-Type: application/json'
```

This endpoint will:
1. Fetch transaction data from Supabase `transaction` table
2. Preprocess and clean the data
3. Train a new neural network model
4. Save model artifacts (model, preprocessor, label encoder)
5. Return training metrics and performance summary

## Database Schema

The Supabase `transaction` table should contain columns matching these expected feature names:
- `transaction_amount` (numeric)
- `carbon_volume` (numeric) 
- `price_per_ton` (numeric)
- `origin_country` (text)
- `cross_border_flag` (boolean/integer)
- `buyer_industry` (text)
- `sudden_transaction_spike` (boolean/integer)
- `transaction_hour` (integer)
- `entity_type` (text)
- `label` (text) - Target labels: "Verified", "Caution", "High-Risk"

## Model Classes

The model predicts three classes:
- **Verified**: Legitimate carbon credit transactions
- **Caution**: Transactions requiring review  
- **High-Risk**: Potentially fraudulent transactions

## Production Deployment

For production, use a WSGI server such as Gunicorn:
```bash
gunicorn --bind 0.0.0.0:8080 app:app
```

## Cloud Storage Support

The training endpoint supports automatic upload to Google Cloud Storage when not in local mode.

### Configuration Options

**Option 1 - Using ARTIFACTS_GCS_PREFIX (Recommended)**:
```bash
LOCAL=false
ARTIFACTS_GCS_PREFIX=gs://your-bucket/models
GOOGLE_CLOUD_PROJECT=your-project-id
```

**Option 2 - Individual URIs**:
```bash
LOCAL=false
MODEL_GCS_URI=gs://your-bucket/models/carbon_watch_model.keras
PREPROCESSOR_GCS_URI=gs://your-bucket/models/preprocessor.pkl
LABEL_ENCODER_GCS_URI=gs://your-bucket/models/label_encoder.pkl
GOOGLE_CLOUD_PROJECT=your-project-id
```

### Authentication

For Google Cloud Storage access, ensure one of the following:

1. **Service Account Key** (for local development):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

2. **Application Default Credentials** (for Cloud Run/GCE):
   ```bash
   gcloud auth application-default login
   ```

3. **Service Account attached to Cloud Run** (for production)

### Training with Cloud Storage

When `LOCAL=false`, the training endpoint will:
1. Train the model locally
2. Save artifacts locally (temporary)
3. Upload artifacts to Google Cloud Storage
4. Update the response with both local and cloud paths

Example response with cloud storage:
```json
{
  "status": "success",
  "model_artifacts": {
    "local_paths": {
      "model_path": "./carbon_watch_model.keras",
      "preprocessor_path": "./preprocessor.pkl",
      "label_encoder_path": "./label_encoder.pkl"
    },
    "cloud_paths": {
      "./carbon_watch_model.keras": "gs://bucket/models/carbon_watch_model.keras",
      "./preprocessor.pkl": "gs://bucket/models/preprocessor.pkl",
      "./label_encoder.pkl": "gs://bucket/models/label_encoder.pkl"
    },
    "storage_mode": "cloud"
  }
}
```

For prediction endpoint, set `LOCAL=false` and configure GCS URIs to load model artifacts from Google Cloud Storage.

