# Titanic Survivors Prediction

This project predicts whether a passenger survived the Titanic disaster based on their features such as age, gender, class, and more. It involves cleaning data, engineering features, building machine learning models, and evaluating the model performance.

---

## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Steps Followed](#steps-followed)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)

---

## Objective

To predict whether a Titanic passenger survived based on available features using machine learning models, and identify the best-performing model.

---

## Dataset

The dataset contains information about Titanic passengers with the following columns:

- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Fare paid by the passenger
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Steps Followed

### 1. Define the Problem

The goal is to build a binary classification model to predict survival (0 or 1).

### 2. Data Cleaning

- Filled missing `Age` values with the median.
- Dropped the `Cabin` column due to many missing values.
- One-hot encoded categorical columns like `Sex` and `Embarked`.

### 3. Exploratory Data Analysis (EDA)

- Visualized survival distribution based on key features.
- Analyzed feature correlations.

### 4. Feature Engineering & Data Preparation

- Selected relevant features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
- Standardized numerical columns (`Age`, `Fare`).

### 5. Splitting Data

- Split the dataset into 80% training and 20% testing sets.

### 6. Model Building

- Trained two models: Logistic Regression and Decision Tree Classifier.
- Compared their performance based on accuracy and classification metrics.

### 7. Model Evaluation

- Evaluated models using classification reports, confusion matrices, and visualizations.

### 8. Model Saving

- Saved the best-performing model using `joblib`.

### 9. Model Testing

- Tested the saved model with sample data to verify predictions.

---

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Model Performance

### Logistic Regression:

- Accuracy: **81.56%**
- Classification Report: Balanced precision, recall, and F1-scores for both classes.

### Decision Tree:

- Accuracy: **79.22%**
- Classification Report: Lower generalization compared to Logistic Regression.

### Selected Model:

**Logistic Regression** was selected as the final model due to better overall performance.

---

## How to Run the Project

1. Clone this repository:

   ```bash
   gh repo clone SidaparaVasu/Cloudcredits-Internship/Titanic_Survivors_Prediction
   ```

2. Navigate to the project directory:

   ```bash
   cd Titanic_Survivors_Prediction
   ```

3. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook to train and evaluate models:

   ```bash
   jupyter notebook titanic_surviors_prediction.ipynb
   ```

5. Test the saved model:

   ```python
   from joblib import load

   model = load('best_model.pkl')
   sample_data = [[3, 0, 22, 1, 0, 7.25, 0, 1]]  # Replace with your test data
   prediction = model.predict(sample_data)
   print("Survived" if prediction[0] == 1 else "Did not survive")
   ```

---

## Results

The Logistic Regression model demonstrated the best performance with 81.56% accuracy. Key insights:

- Passengers in 1st class had a higher survival rate.
- Females were more likely to survive compared to males.
- Younger passengers had better survival chances.

---

## Future Enhancements

- Use advanced algorithms like Random Forest or Gradient Boosting for better accuracy.
- Implement cross-validation to avoid overfitting.
- Deploy the model using a Flask or Django web application.

---

## Author

Project developed by **Vasu Sidapara**.
