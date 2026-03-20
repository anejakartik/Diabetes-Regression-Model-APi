# EAI6010 Assignment: Model Microservice Deployment

## 1) Service input and output (general description)

This microservice exposes a **PyTorch regression model** adapted from my previous assignment: *off_the_shelf_model_diabetes_regression*.

- **Input:** JSON object with 10 numeric features from the Diabetes dataset schema:
  - `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`
- **Output:** JSON object with:
  - Echoed input
  - `predicted_disease_progression` (continuous numeric prediction)
  - Unit description text

## 2) Specific examples of input and output

### Example Request

POST `/predict`

```json
{
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
}
```

### Example Response

```json
{
  "input": {
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
  },
  "predicted_disease_progression": 183.551,
  "units": "quantitative progression score (higher means worse progression)"
}
```

## 3) Service URL

- Local service URL: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`
- Public deployment URL (for instructor grading): **ADD YOUR RENDER URL HERE**

## 4) Instructor-friendly invocation instructions

1. Open API docs in browser: `/docs`
2. Expand `POST /predict`
3. Click **Try it out**
4. Paste the sample JSON above
5. Click **Execute**
6. Verify HTTP 200 and prediction in response body

## 5) Availability plan until grading is complete

- Deploy this app on Render using the included `render.yaml` and `Dockerfile`.
- Keep auto-deploy enabled.
- Do not delete the service until final grading is posted.
- Verify health endpoint once per day: `/health`
