# Diabetes Prediction Project

## Overview
This project focuses on predicting diabetes using machine learning techniques. It implements and compares different classification algorithms including Support Vector Machines (SVM) and Logistic Regression on the Pima Indians Diabetes Dataset.

## Dataset
The project uses the Pima Indians Diabetes Dataset, which contains diagnostic measurements for 768 patients. The dataset includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours after an oral glucose tolerance test)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
- **Age**: Age in years
- **Outcome**: Class variable (0: non-diabetic, 1: diabetic)

## Models Implemented

### 1. Support Vector Machine (SVM)
Implemented in `diabetes_prediction_svm.ipynb`:
- Data preprocessing and standardization
- Feature analysis and visualization
- Model training using SVM with GridSearchCV for hyperparameter tuning
- Model evaluation using accuracy metrics, confusion matrix, and classification report

### 2. Logistic Regression
Implemented in `logistic_regression_analysis_python.ipynb`:
- Custom implementation of logistic regression from scratch
- Comparison with scikit-learn's LogisticRegression and SGDClassifier
- Performance evaluation and comparison between the implementations
- Visualization of training accuracy across epochs

## Results

### SVM Model
- Training Accuracy: ~79.3%
- Test Accuracy: ~77.1%
 [(LR_IMG.png)]
### Logistic Regression Models
- Scratch Implementation Accuracy: ~73.4%
- Scikit-learn Implementation Accuracy: ~77.3%
- SGDClassifier with Log Loss Accuracy: ~70.8%
[(SVM_IMG.png)]
## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone this repository
2. Ensure you have all the required libraries installed
3. Run the Jupyter notebooks:
   - `diabetes_prediction_svm.ipynb` for SVM implementation
   - `logistic_regression_analysis_python.ipynb` for Logistic Regression implementation

## Future Work
- Implement additional machine learning algorithms for comparison
- Feature engineering to improve model performance
- Hyperparameter optimization for all models
- Deploy the best performing model as a web application

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements
- The Pima Indians Diabetes Dataset from the UCI Machine Learning Repository
- Scikit-learn documentation and community