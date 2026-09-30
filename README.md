# Insurance Cost Prediction with Linear Regression

A reproducible machine learning project that uses real-world insurance data to explore, train, evaluate, and compare linear regression models in Python.

This project compares two approaches for fitting the same linear regression model:

- Normal Equation — a direct analytical solution
- Gradient Descent — an iterative optimization method

The goal is not only to run a regression model, but also to understand how the model is trained, how different training methods compare, and how to interpret model performance for a non-technical audience.

---

## Project Overview

This project uses an insurance dataset to study the relationship between BMI and medical expenses.

The workflow covers:

1. Data inspection and exploratory analysis
2. Correlation analysis
3. Simple linear regression
4. Training with the Normal Equation
5. Training with Gradient Descent
6. Model evaluation
7. Comparison of both training strategies
8. Business interpretation of the results

The project includes both Python scripts and a Jupyter Notebook so the analysis can be reviewed step by step.

---

## Business Question

Can BMI help explain or predict medical expenses?

The analysis uses BMI as the main input feature and medical expenses as the target variable.

This is intentionally a simple one-feature model so that the training process can be understood clearly before moving to more complex models.

---

## Methods

### Pearson Correlation

Pearson correlation is used as an exploratory step to measure the direction and strength of the linear relationship between BMI and medical expenses.

Correlation describes association, not causation.

### Simple Linear Regression

The model has the form:

predicted expenses = w0 + w1 × BMI

Where:

- w0 = intercept
- w1 = slope
- BMI = input feature
- predicted expenses = model prediction

### Normal Equation

The Normal Equation calculates the regression coefficients directly using a mathematical closed-form solution.

It does not require:

- a learning rate
- epochs
- repeated weight updates

This approach is useful when the number of features is relatively small.

### Gradient Descent

Gradient Descent starts with initial values for the model weights and repeatedly updates them in the direction that reduces prediction error.

The basic process is:

Predict → Calculate Error → Calculate Gradient → Update Weights → Repeat

The learning rate controls the size of each update step.

### Why Compare Both?

Both approaches are trying to solve the same linear regression problem.

The Normal Equation solves directly for the coefficients, while Gradient Descent approaches the solution iteratively.

Comparing them helps demonstrate that different training strategies can produce very similar fitted models.

---

## Model Evaluation

The project uses several evaluation metrics.

### Residual

Residual = Actual Value − Predicted Value

A residual shows how far one individual prediction is from the observed value.

### MSE

Mean Squared Error measures the average squared prediction error.

Lower MSE indicates smaller prediction errors when comparing models on the same target and dataset.

### RMSE

Root Mean Squared Error converts MSE back into the original target units, making the error easier to interpret.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

### R²

R² measures how much of the observed variation in the target is explained by the fitted model.

A low R² does not necessarily mean the optimization algorithm failed. It may mean that the selected feature contains limited predictive information.

---
## Results

Using the same training data, the Normal Equation and Gradient Descent produced the same fitted regression model:

**Predicted Expenses = 1216.99 + 400.61 × BMI**

Both methods produced the same evaluation results:

- Train MSE: 143,772,403.04
- Test MSE: 129,028,708.19
- Train RMSE: $11,990.51
- Test RMSE: $11,359.08
- Train MAE: $9,339.00
- Test MAE: $8,924.52
- Train R²: 0.0406
- Test R²: 0.0226

The slope indicates that a one-unit increase in BMI is associated with an increase of approximately $400.61 in predicted medical expenses.

However, the test R² of 0.0226 shows that BMI alone explains only about 2.26% of the observed variation in medical expenses.

Gradient Descent reduced the loss substantially and converged to the same coefficients as the Normal Equation, demonstrating that both training strategies can reach the same least-squares solution.

The model performs slightly better than a naive mean-prediction baseline, but its predictive power remains limited. Additional features such as age, smoking status, and region would likely be necessary for a stronger predictive model.


## Key Analytical Takeaways

This project demonstrates several important machine learning lessons:

- Correlation should be examined before fitting a linear model.
- Correlation does not prove causation.
- The Normal Equation and Gradient Descent can solve the same regression problem using different approaches.
- Feature standardization helps Gradient Descent train more smoothly.
- A model can converge successfully while still having limited predictive power.
- Model performance should be interpreted in the context of the business problem rather than judged from one metric alone.

---

## Repository Structure

```text
insurance-cost-linear-regression/
│
├── data/
│   ├── insurance-premium-prediction/
│   │   └── insurance.csv
│   └── model_comparison_bmi_expenses.png
│
├── 01_simple_linear.py
├── 02_ols_normal_equation.py
├── 03_gradient_descent.py
├── 04_compare_models_visual.py
├── linear_regression_lab.ipynb
├── requirements.txt
├── .gitignore
└── README.md
