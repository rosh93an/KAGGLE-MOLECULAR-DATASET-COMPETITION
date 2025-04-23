# KAGGLE-MOLECULAR-DATASET-COMPETITION
 Molecular Photostability Prediction (T80)
This repository presents a machine learning solution to predict the photostability lifetime (T80) of organic molecules. The work is based on data provided by the NSF Molecule Maker Lab Institute and was developed as part of a competition aimed at discovering the most informative molecular features using data-driven models.

📈 Final Score: MSLE = 0.39
🚀 Problem Statement
Predicting the photostability of molecules is crucial for designing long-lasting organic materials (e.g., solar cells). Given a dataset of ~150 calculated molecular features and SMILES strings for ~100 molecules, the goal is to:

Select informative features that influence T80

Train a regression model to generalize on unseen molecules

Evaluate performance using Mean Squared Log Error (MSLE)

🧬 Approach
Feature Engineering: RDKit-derived descriptors (MolWt, HDonors, TPSA, etc.)

Feature Selection: SelectKBest with k=25

Modeling: XGBRegressor (XGBoost) with hyperparameter tuning

Validation: 4-fold cross-validation

No pretrained models used — this is a traditional feature-based approach

📁 Files
notebook.ipynb — Main workflow: preprocessing, training, validation, submission

final_submission.csv — Predictions on test molecules

xgb_best_model_k25.pkl — Saved model using joblib

requirements.txt — Dependencies


import joblib
model = joblib.load("xgb_best_model_k25.pkl")
Predict on test data:

python
Copy
Edit
preds = model.predict(X_test_selected)  # X_test_selected: preprocessed test features
📚 Acknowledgements
RDKit for molecular feature computation

NSF Molecule Maker Lab Institute

scikit-learn & XGBoost
