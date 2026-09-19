# Loan Approval Prediction Using Artificial Neural Network

Predicting loan approval from a 4,269-row dataset using a sequential Artificial Neural Network built with TensorFlow/Keras.

## Dataset
- 4,269 rows with 13 features (loan amount, income, CIBIL score, etc.)

## Project Workflow
1. Data cleaning and preprocessing (missing values, one-hot encoding, StandardScaler)
2. Exploratory Data Analysis (EDA) with histograms and correlation heatmap
3. Built a sequential ANN model (128 → 64 → 1) with ReLU and sigmoid activation
4. Evaluated model performance with confusion matrix and classification report

## Results
- **Test Accuracy:** 96.5%
- **Precision:** 0.965
- **Recall:** 0.941
- **F1-Score:** 0.953

## Tools Used
- Python (pandas, numpy, scikit-learn)
- TensorFlow / Keras
- Matplotlib, Seaborn
- Google Colab

## Files
- `loan-approval-prediction.ipynb` — Complete notebook with code and outputs
