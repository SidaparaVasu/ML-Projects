# Boston Housing Price Prediction

This project demonstrates the prediction of house prices in the Boston area using machine learning. The dataset contains various features such as crime rate, number of rooms, and proximity to employment centers. The primary goal is to build and evaluate a regression model that accurately predicts house prices.

---

## Project Overview

- **Dataset**: Boston Housing Dataset
- **Objective**: Predict house prices (`MEDV`) based on features like:
  - Crime rate (`CRIM`)
  - Number of rooms (`RM`)
  - Tax rate (`TAX`)
  - Distance to employment centers (`DIS`)
  - ... and other features.
- **Algorithm**: Linear Regression, Ridge Regression, Lasso Regression
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R² Score

---

## Steps Followed

### 1. Data Preprocessing

- Loaded the Boston Housing dataset.
- Handled missing values by imputing medians.
- Scaled features using StandardScaler.
- Checked for multicollinearity using Variance Inflation Factor (VIF).

### 2. Exploratory Data Analysis (EDA)

- Visualized relationships between features and house prices.
- Correlation matrix to identify strongly correlated features.
- Histogram and scatter plots for feature distributions.

### 3. Model Development

- Split data into training (80%) and testing (20%) sets.
- Trained models:
  - Linear Regression
  - Ridge Regression (L2 Regularization)
  - Lasso Regression (L1 Regularization)
- Evaluated models using MSE and R² Score.

### 4. Model Refinements

- Performed hyperparameter tuning for Ridge and Lasso using GridSearchCV.
- Applied cross-validation to ensure model stability.

### 5. Model Interpretation

- Analyzed model coefficients to determine feature importance.
- Residual analysis to check for model assumptions.
- Visualized actual vs predicted prices.

### 6. Deployment

- Saved the best-performing model (Ridge) using joblib.
- Created a Python function to load the model and make predictions on new data.

---

## Performance Comparison

| Model             | MSE   | R² Score |
| ----------------- | ----- | -------- |
| Linear Regression | 25.02 | 0.66     |
| Ridge Regression  | 25.02 | 0.66     |
| Lasso Regression  | 25.04 | 0.66     |

## Lasso Regression gives best results.

## File Structure

```plaintext
.
├── bostan_housing_price_prediction.ipynb  # Jupyter Notebook with code
├── best_house_price_model.pkl             # Saved Ridge regression model
├── HousingData.csv                        # If Dataset doesn't load, we will use this file
├── README.md                              # Project documentation
```

## How to Use

1. Clone the repository
```
git clone [<repository_url>](https://github.com/SidaparaVasu/Cloudcredits-Internship/tree/main/Boaston_House_Price_Prediction)
cd Boaston_House_Price_Prediction
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the notebook to explore the project or make predictions:
```
jupyter notebook bostan_housing_price_prediction.ipynb
```

4. Use the saved model (best_house_price_model.pkl) to predict prices with your own data:
```
from joblib import load

model = load("best_house_price_model.pkl")
example_features = [0.02, 0, 7.07, 0, 0.469, 6.5, 68, 3.5, 2, 300, 15, 390, 10]
predicted_price = model.predict([example_features])
print(predicted_price)
```

## Technologies Used

• Programming Language: Python
• Libraries:
    • pandas
    • numpy
    • matplotlib
    • seaborn
    • sklearn
    • joblib


## Future Work
• Add more advanced regression models like Decision Trees or Random Forests.
• Deploy the model as a web application using Flask or FastAPI.
• Enhance feature engineering for better insights.


Author
Vasu Sidapara
For queries, reach out at [vasupatel303@gmail.com].
