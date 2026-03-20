# EAI6010 Model Microservice (Diabetes Regression)

This folder contains a FastAPI microservice that exposes the model from the previous assignment (`off_the_shelf_model_diabetes_regression.ipynb`).

## Endpoints

- `GET /` service metadata
- `GET /health` readiness + model metrics
- `POST /predict` regression prediction
- `GET /docs` interactive Swagger UI

## 1) Local run (quick)

```bash
cd /Users/kartik/mps/EAI6010/Diabetes-Regression-Model-APi
/Users/kartik/mps/EAI6010/.venv/bin/python train_and_export_model.py
/Users/kartik/mps/EAI6010/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open:
- http://127.0.0.1:8000/docs

## 2) Test request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.038076,
    "sex": 0.05068,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034821,
    "s3": -0.043401,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646
  }'
```

## 3) Deploy on Render (public URL)

1. Push this folder to a GitHub repo.
2. In Render, click **New +** -> **Blueprint**.
3. Select the repo and deploy using `render.yaml`.
4. After deploy is healthy, copy your public URL.
5. Put that URL in `ASSIGNMENT_MICROSERVICE_REPORT.md`.
6. Keep the service running until the grade is posted.

## 4) Instructor invocation instructions

1. Open `<PUBLIC_URL>/docs`
2. Expand `POST /predict`
3. Click **Try it out**
4. Paste sample JSON
5. Click **Execute**
6. Confirm HTTP 200 and returned prediction
