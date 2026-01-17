# SCIENTIFIC PROGRAMMING - FINAL PROJECT
**GROUP G**
> - Ana Diaz Acevedo
> - Carla Aullón Coral
> - Carmen Aznar Mathonneau
> - Julen Rodriguez Eraso
> - Ludovic Mean Touroyan
> - Serina Dhruvlata Allen
> - Victor Gutierrez Gonzalez

<br>

## Project Overview
This project aims to develop a **robust and accurate predictive model** to assist in the diagnosis of breast cancer based on specific clinical and morphological markers. Using the *Breast Cancer Wisconsin (Diagnostic)* dataset, we apply machine learning and biostatistical techniques to classify tumors as **benign** or **malignant**.
The final outcome of the project is a trained predictive model encapsulated within an **API**, allowing users to input patient marker values and receive a diagnostic prediction.

---
## Dataset
* **Name:** Breast Cancer Wisconsin (Diagnostic)
* **Source:** UCI Machine Learning Repository
* **Link:** [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
* **Samples:** 569
* **Features:** 30 numerical features derived from digitized images of fine needle aspirates (FNAs) of breast masses
* **Target Variable:** Diagnosis
  * 1 = Malignant
  * 0 = Benign

---

## Project Objectives

* Perform exploratory data analysis (EDA) to understand feature distributions and relationships
* Preprocess the dataset (handling scaling, feature selection, etc.)
* Train and evaluate multiple machine learning models
* Select the best-performing model based on appropriate evaluation metrics
* Deploy the trained model through an API interface
* Ensure reproducibility and collaborative development using Git and GitHub

---

## Repository Structure

```
SP-GroupG-Project/
│
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned and preprocessed data
│
├── notebooks/         # Jupyter notebooks for EDA and modeling
│
├── src/               # Core Python modules
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── api/               # API implementation
│   └── app.py
│
├── requirements.txt   # Project dependencies
├── README.md          # Project documentation
```

---

## Tools & Technologies

* **Programming Language:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * matplotlib, seaborn
  * FastAPI or Flask (for API deployment)

* **Version Control:** Git & GitHub

---

## Development Workflow

* Each team member works on a **separate Git branch** to ensure individual contributions are tracked
* Branches are merged into `main` via Pull Requests
* Commit messages are written clearly to describe changes

---

## Model Evaluation

Models are evaluated using appropriate classification metrics, such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Special attention is given to **recall**, as minimizing false negatives is critical in medical diagnosis.

---

## API Description (Planned)

The API will:

* Accept input features corresponding to the dataset markers
* Apply the same preprocessing pipeline used during training
* Return a predicted diagnosis (`Benign` or `Malignant`)

---

###### Logistic Regression ######
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# ---- Path corrected for your machine ----
DATA_PATH = r"C:\Users\Admin\Desktop\wdbc.data"

feature_names = [
    "mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",
    "mean_compactness","mean_concavity","mean_concave_points","mean_symmetry","mean_fractal_dimension",
    "radius_error","texture_error","perimeter_error","area_error","smoothness_error",
    "compactness_error","concavity_error","concave_points_error","symmetry_error","fractal_dimension_error",
    "worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness",
    "worst_compactness","worst_concavity","worst_concave_points","worst_symmetry","worst_fractal_dimension"
]
columns = ["subject_id", "diagnosis"] + feature_names

# Safety check
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, header=None, names=columns)

print("Dataset shape:", df.shape)
df.head()
# ======================================================
# 2) Remap diagnosis (B=0 benign, M=1 malignant)
# ======================================================
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

print("Diagnosis distribution (0=benign, 1=malignant):")
print(df["diagnosis"].value_counts())


# ======================================================
# 3) Prepare X and y
# ======================================================
X = df.drop(columns=["subject_id", "diagnosis"])
y = df["diagnosis"]


# ======================================================
# 4) Feature selection (aligned with report)
#    - Drop highly correlated features (abs corr > 0.90)
#    - Drop weak predictors vs diagnosis (abs corr < 0.05)
# ======================================================
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > 0.90)
]

corr_with_target = X.join(y).corr(numeric_only=True)["diagnosis"].drop("diagnosis").abs()
low_corr_features = corr_with_target[corr_with_target < 0.05].index.tolist()

features_to_drop = sorted(set(high_corr_features + low_corr_features))

print("\n--- Feature Selection ---")
print(f"Highly correlated features dropped (>0.90): {len(high_corr_features)}")
print(high_corr_features)
print(f"\nLow-correlation features dropped (<0.05): {len(low_corr_features)}")
print(low_corr_features)
print(f"\nTotal dropped: {len(features_to_drop)}")

X_selected = X.drop(columns=features_to_drop)
print("Remaining features:", X_selected.shape[1])


# ======================================================
# 5) Normalisation (MinMaxScaler)
# ======================================================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_selected)


# ======================================================
# 6) Train / Test split
# ======================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train diagnosis counts:\n", y_train.value_counts())
print("Test diagnosis counts:\n", y_test.value_counts())


# ======================================================
# 7) KNN model (for comparison)
# ======================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\n================ KNN RESULTS ================")
print(f"Accuracy: {acc_knn:.4f}")
print(f"Recall:   {rec_knn:.4f}")
print("Confusion matrix:\n", cm_knn)
print("\nClassification report:\n", classification_report(y_test, y_pred_knn))


# ======================================================
# 8) Logistic Regression (YOUR PART - Step 4)
# ======================================================
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\n=========== LOGISTIC REGRESSION RESULTS ===========")
print(f"Accuracy: {acc_lr:.4f}")
print(f"Recall:   {rec_lr:.4f}")
print("Confusion matrix:\n", cm_lr)
print("\nClassification report:\n", classification_report(y_test, y_pred_lr))


# ======================================================
# 9) Final comparison summary
# ======================================================
print("\n================ FINAL COMPARISON ================")
print(f"KNN  -> Accuracy: {acc_knn:.4f} | Recall: {rec_knn:.4f}")
print(f"LOGR -> Accuracy: {acc_lr:.4f} | Recall: {rec_lr:.4f}")


## Contributors

* Team members contribute via individual branches
* All contributions are documented through Git commit history and pull requests

