CarbonWatch Backend

This small Flask backend exposes a single prediction endpoint that loads the trained Keras model and preprocessing artifacts from the `ML/` folder.

Files expected in the repository:
- `ML/carbon_watch_model.keras`
- `ML/preprocessor.pkl`
- `ML/label_encoder.pkl`

Quick start (from repository root):

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Backend/requirements.txt
```

2. Run the app (development mode):

```bash
python Backend/app.py
```

3. Example request (single record):

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"feature1": 1.0, "feature2": "A", "feature3": 42}'
```

Or send multiple instances:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"feature1": 1.0, "feature2": "A"}, {"feature1": 2.0, "feature2": "B"}]}'
```

Notes:
- The feature names in the JSON must match the columns used during training (the CSV columns excluding `Label`).
- For production, use a WSGI server such as Gunicorn.
