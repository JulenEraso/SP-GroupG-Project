from fastapi import FastAPI, Query
# from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse, HTMLResponse
import pandas as pd
import joblib

# Threshold for classifying tumor as Malignant (1) or Benign (0)
THRESHOLD = 0.4703

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="API for breast cancer prediction using a logistic regression model"
)

# Load the trained pipeline (includes preprocessing: mean imputation + MinMax scaling)
pipeline = joblib.load("model_pipeline.pkl")

@app.get("/", response_class=HTMLResponse, tags = ["Overview"])
def home():
    """
    This endpoint gives a brief overview of the API in HTML format for users.
    """
    return """
    <html>
        <head>
            <title>Breast Cancer Diagnosis API</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 60px;">
            
            <h1 style="color: #8B0000;">Breast Cancer Diagnosis API</h1>

            <h3>Project Overview</h3>
            <p style="width: 60%; margin: auto;">
                This project aims to develop a robust and accurate predictive model
                to assist in the diagnosis of breast cancer based on specific clinical
                and morphological markers. Using the Breast Cancer Wisconsin (Diagnostic)
                dataset, a logistic regression is applied to 
                classify tumors as benign or malignant.
            </p>

            <br>

            <h3>Training Dataset</h3>
            <p>Breast Cancer Wisconsin (Diagnostic)</p>

            <h3>Output</h3>
            <p>Binary diagnosis (0 = Benign, 1 = Malignant)</p>

            <h3>Usage</h3>
            <p style="width: 60%; margin: auto;">
                Provide 17 numeric patient markers corresponding to cell morphology 
                measurements from a Fine Needle Aspirate (FNA) of the breast mass. 
                Submit the values via a POST request to the <b>/predict</b> endpoint 
                using the interactive API documentation below.
            </p>
            <br>

            <a href="/docs" style="font-size: 18px; color: blue;">
                👉 Go to Interactive API Documentation to try it
            </a>

        </body>
    </html>
    """

@app.post("/predict", response_class=PlainTextResponse, tags=["Prediction"], 
          summary = "Predict diagnosis"
)
def predict(    
    mean_radius: float = None,
    mean_texture: float = None,
    mean_smoothness: float = None,
    mean_compactness: float = None,
    mean_concavity: float = None,
    mean_symmetry: float = None,
    mean_fractal_dimension: float = None,
    radius_error: float = None,
    compactness_error: float = None,
    concavity_error: float = None,
    concave_points_error: float = None,
    worst_smoothness: float = None,
    worst_compactness: float = None,
    worst_concavity: float = None,
    worst_concave_points: float = None,
    worst_symmetry: float = None,
    worst_fractal_dimension: float = None
    ):
    """
    Predicts whether a breast tumor is likely benign or malignant based on 17 cell 
    morphology markers obtained from a Fine Needle Aspirate (FNA) of the breast mass.

    Each input corresponds to a feature calculated across all cell nuclei in the sample:

    Feature types:
    - `mean_...` : the average value of the feature across all cells.
    - `..._error` : the standard deviation of the feature across cells (measure of variability).
    - `worst_...` : the most extreme (highest or most concerning) value of the feature among all cells.

    Feature descriptions:
    - `radius` : mean of distances from center to points on the perimeter
    - `texture` : standard deviation of gray-scale values
    - `perimeter` : cell nucleus perimeter
    - `area` : cell nucleus area
    - `smoothness` : local variation in radius lengths
    - `compactness` : perimeter^2 / area - 1.0
    - `concavity` : severity of concave portions of the contour
    - `concave points` : number of concave portions of the contour
    - `symmetry` : symmetry of the cell nucleus
    - `fractal dimension` : "coastline approximation" - 1

    Providing these 17 markers allows the trained model pipeline to return a prediction 
    for the tumor’s likely diagnosis.
    """

    column_names = [
    "mean_radius", "mean_texture", "mean_smoothness", "mean_compactness",
    "mean_concavity", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "compactness_error", "concavity_error", "concave_points_error",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
    ]

    # Convert user inputs into a DataFrame to feed into the pipeline
    X_df = pd.DataFrame([[
        mean_radius, mean_texture, mean_smoothness, mean_compactness, 
        mean_concavity, mean_symmetry, mean_fractal_dimension, radius_error,
        compactness_error, concavity_error, concave_points_error, 
        worst_smoothness, worst_compactness, worst_concavity,
        worst_concave_points, worst_symmetry, worst_fractal_dimension
    ]], columns=column_names)
    
    # Get probability of malignancy from the trained pipeline
    proba = pipeline.predict_proba(X_df)[0][1]

    # Apply chosen best threshold to classify tumor
    prediction = int(proba >= THRESHOLD) 
    # prediction = pipeline.predict(X)[0]

    dic_res = {0: "Benign", 1: "Malignant"}
    return f"The tumor is classified by the model as: {dic_res[prediction]}"
