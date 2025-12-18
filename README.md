# SCIENTIFIC PROGRAMMING - FINAL PROJECT
**GROUP G**
> - Ana Diaz Acevedo
> - Carla Aullón Coral
> - Carmen Aznar Mathonneau
> - Julen Rodriguez Eraso
> - Ludovic Mean Touroyan
> - Serina Allen
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

## Contributors

* Team members contribute via individual branches
* All contributions are documented through Git commit history and pull requests

