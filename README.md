# Insurance Cost Prediction with Linear Regression

A reproducible machine learning project using real-world insurance data to explore, train, evaluate, and compare linear regression models in Python.

This project compares two approaches for fitting the same linear regression model:

- Normal Equation — a direct analytical solution
- Gradient Descent — an iterative optimization method

The goal is not only to run linear regression, but also to understand how the model is trained, how different training strategies compare, and how to interpret model performance for a non-technical audience.

---

## Project Overview

This project uses an insurance dataset to study the relationship between BMI and medical expenses.

The workflow includes:

1. Data inspection and exploratory analysis
2. Correlation analysis
3. Simple linear regression
4. Normal Equation training
5. Gradient Descent training
6. Model evaluation
7. Comparison of both training strategies
8. Business interpretation

The project includes Python scripts and a Jupyter Notebook so the analysis can be reviewed step by step.

---

## Business Question

Can BMI help explain or predict medical expenses?

BMI is used as the main input feature, and medical expenses are used as the target variable.

This is intentionally a simple one-feature model so the training process can be understood clearly before moving to more complex models.

---

## Methods

### Pearson Correlation

Pearson correlation is used as an exploratory step to examine the direction and strength of the linear relationship between BMI and medical expenses.

Correlation shows association, not causation.

### Simple Linear Regression

The model has the form:

```text
Predicted Expenses = w0 + w1 × BMI
```

Where:

- `w0` = intercept
- `w1` = slope
- BMI = input feature
- Predicted Expenses = model prediction

### Normal Equation

The Normal Equation calculates the regression coefficients directly.

It does not require:

- learning rate
- epochs
- repeated weight updates

It is convenient when the number of features is relatively small.

### Gradient Descent

Gradient Descent begins with initial model weights and repeatedly updates them to reduce prediction error.

The basic process is:

```text
Predict
↓
Calculate Error
↓
Calculate Gradients
↓
Update Weights
↓
Repeat
```

The learning rate controls the size of each update step.

### Why Compare Both?

Both approaches solve the same linear regression problem.

The Normal Equation solves directly for the coefficients, while Gradient Descent approaches the solution iteratively.

---

## Results

Using the same training data, the Normal Equation and Gradient Descent produced the same fitted regression model:

```text
Predicted Expenses = 1216.99 + 400.61 × BMI
```

Model evaluation results:

| Metric | Normal Equation | Gradient Descent |
|---|---:|---:|
| Train MSE | 143,772,403.04 | 143,772,403.04 |
| Test MSE | 129,028,708.19 | 129,028,708.19 |
| Train RMSE | $11,990.51 | $11,990.51 |
| Test RMSE | $11,359.08 | $11,359.08 |
| Train MAE | $9,339.00 | $9,339.00 |
| Test MAE | $8,924.52 | $8,924.52 |
| Train R² | 0.0406 | 0.0406 |
| Test R² | 0.0226 | 0.0226 |

The slope indicates that a one-unit increase in BMI is associated with an increase of approximately $400.61 in predicted medical expenses.

The test R² of 0.0226 means that BMI alone explains only about 2.26% of the observed variation in medical expenses.

Gradient Descent reduced the loss substantially and converged to the same coefficients as the Normal Equation.

This shows that both training strategies can reach the same least-squares solution.

The model performs slightly better than a simple mean-prediction baseline, but its predictive power remains limited.

Additional predictors such as age, smoking status, and region may improve model performance.

## Two-Feature Extension: BMI + Age

This project was extended from a one-feature BMI model to a two-feature model using BMI and age.

The extended model is:

```text
expenses = w0 + w1*bmi + w2*age
```

Both the Normal Equation and Gradient Descent produced nearly identical coefficients:

- Intercept (w0): -6437.35
- BMI coefficient (w1): 333.39
- Age coefficient (w2): 241.90
- R²: 0.1173

The BMI-only baseline had an R² of 0.0394, so adding age improved explanatory power by about 7.78 percentage points.

The 0.0394 baseline is computed on the full dataset without a train/test split, to match the reference implementation in `02_ols_normal_equation.py`. The 0.0406 and 0.0226 figures reported earlier come from the train/test split used in `04_compare_models_visual.py`.

The detailed comparison results are exported to:

```text
reports/assignment_results.csv
```

The written interpretation is available in:

```text
analysis_memo.md
```

---

## Key Analytical Takeaways

- Correlation should be examined before fitting a linear model.
- Correlation does not prove causation.
- Normal Equation and Gradient Descent can solve the same regression problem using different approaches.
- Feature standardization improves Gradient Descent stability.
- Successful optimization does not automatically mean strong predictive performance.
- Model performance should be interpreted in the context of the business problem.
- BMI alone is not sufficient for strong medical-expense prediction.

---

## Repository Structure

```text
insurance-cost-linear-regression/
│
├── data/
│   ├── insurance-premium-prediction/
│   │   └── insurance.csv                   # dataset (committed, no download needed)
│   └── model_comparison_bmi_expenses.png   # output of 04
│
├── reports/
│   └── assignment_results.csv              # output of the two-feature script
│
├── 01_simple_linear.py                     # course material — data exploration
├── 02_ols_normal_equation.py               # course material — analytical OLS
├── 03_gradient_descent.py                  # course material — iterative OLS
├── 04_compare_models_visual.py             # course material — method comparison
│
├── two_feature_insurance_regression.py     # my extension — BMI + age model
├── analysis_memo.md                        # my written interpretation
│
├── linear_regression_lab.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Setup Instructions for Beginners

These instructions assume that the user has little or no previous experience with Python, Git, or Visual Studio Code.

Instructions are provided for both Windows and macOS.

---

## Step 1 — Install Python

Go to:

https://www.python.org/downloads/

Download Python 3.10 or newer.

### Windows

During installation, make sure to select:

```text
Add Python to PATH
```

Then complete the installation.

### macOS

Download and install the current Python 3 version from the Python website.

---

## Step 2 — Install Visual Studio Code

Go to:

https://code.visualstudio.com/

Download and install Visual Studio Code.

Use the default installation options.

---

## Step 3 — Install Git

Go to:

https://git-scm.com/downloads

Download and install Git.

Use the default installation options.

To confirm Git is installed, open a terminal and type:

```bash
git --version
```

You should see a Git version number.

---

## Step 4 — Install VS Code Extensions

Open Visual Studio Code.

Click the Extensions icon on the left side.

Search for and install:

1. Python — Microsoft
2. Jupyter — Microsoft

---

## Step 5 — Download This Project

Open Visual Studio Code.

From the top menu, choose:

```text
Terminal → New Terminal
```

Copy and run:

```bash
git clone https://github.com/fhu0914/insurance-cost-linear-regression.git
cd insurance-cost-linear-regression
```

The first command downloads the project.

The second command moves the terminal into the project folder.

---

## Step 6 — Create a Virtual Environment

A virtual environment keeps this project's Python packages separate from other projects.

### Windows PowerShell

Run:

```powershell
python -m venv .venv
```

Then activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If Windows displays an error saying scripts are disabled, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

When activation succeeds, the terminal should begin with:

```text
(.venv)
```

### macOS

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When activation succeeds, the terminal should begin with:

```text
(.venv)
```

---

## Step 7 — Install Required Packages

Make sure the virtual environment is activated.

### Windows

Run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

Run:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Wait until installation finishes.

### What Each Script Does

| Script | Purpose | Main Output |
|---|---|---|
| `01_simple_linear.py` | Explore the data: shape, summary statistics, correlations with expenses | Correlation values and text-based histograms |
| `02_ols_normal_equation.py` | Train a single-feature model (BMI) analytically | w0, w1, MSE, R² on the full dataset |
| `03_gradient_descent.py` | Train the same model iteratively, with feature standardization | Loss curve across epochs, final weights in original units |
| `04_compare_models_visual.py` | Compare both methods on a train/test split | Metrics table, scatter plot saved to `data/` |
| `two_feature_insurance_regression.py` | **My extension:** add `age` as a second feature | Comparison table saved to `reports/assignment_results.csv` |

Scripts 01–04 come from the course materials and build up the foundation.
The fifth script is my own extension and is where the assignment deliverable
lives. Each script prints extensive explanatory output by design — this is a
teaching repository, so the terminal output walks through the reasoning rather
than just reporting numbers.

---

## Step 8 — Run the Python Scripts

Run the scripts in this order.

### Windows

```powershell
python 01_simple_linear.py
python 02_ols_normal_equation.py
python 03_gradient_descent.py
python 04_compare_models_visual.py
python two_feature_insurance_regression.py
```

### macOS

```bash
python3 01_simple_linear.py
python3 02_ols_normal_equation.py
python3 03_gradient_descent.py
python3 04_compare_models_visual.py
python3 two_feature_insurance_regression.py
```

The scripts follow this workflow:

```text
01 → Explore and understand the data
02 → Fit linear regression with the Normal Equation
03 → Fit linear regression with Gradient Descent
04 → Compare the two methods
two_feature_insurance_regression.py → Extend to a two-feature model (BMI + age)
```

If the terminal returns to the command prompt without an error, the script completed successfully.

---

## Step 9 — Open the Jupyter Notebook (Optional)

This step is optional. The notebook contains the same analysis in an interactive
format for readers who prefer notebooks over scripts. Everything required to
reproduce the results has already run in Step 8.

In Visual Studio Code, open:

```text
linear_regression_lab.ipynb
```

Look at the upper-right corner of the notebook.

Click:

```text
Select Kernel
```

Select the Python interpreter located inside:

```text
.venv
```

Then click:

```text
Run All
```

You may also run the notebook cells one at a time from top to bottom.

---

## Expected Workflow

```text
Load Data
↓
Inspect Data
↓
Correlation Analysis
↓
Linear Regression
↓
Normal Equation
↓
Gradient Descent
↓
Predictions
↓
Residuals
↓
MSE / RMSE / MAE / R²
↓
Model Comparison
↓
Business Interpretation
```

---

## Expected Main Result

Both training methods should produce very similar or identical coefficients when they are applied to the same training data.

The comparison script should report approximately:

```text
Predicted Expenses = 1216.99 + 400.61 × BMI
Test R² = 0.0226
Test RMSE = $11,359.08
```

The comparison chart will be saved to:

```text
data/model_comparison_bmi_expenses.png
```

---

## Troubleshooting

### Error: ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'pandas'
```

Make sure the virtual environment is activated.

Then reinstall the required packages.

Windows:

```powershell
python -m pip install -r requirements.txt
```

macOS:

```bash
python3 -m pip install -r requirements.txt
```

---

### PowerShell Says Scripts Are Disabled

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### VS Code Uses the Wrong Python Interpreter

Open the Command Palette.

Windows:

```text
Ctrl + Shift + P
```

macOS:

```text
Command + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Choose the Python interpreter inside `.venv`.

---

### Jupyter Notebook Does Not Run

Open:

```text
linear_regression_lab.ipynb
```

Click:

```text
Select Kernel
```

Choose the `.venv` Python environment.

---

### Git Command Is Not Recognized

Close and reopen Visual Studio Code after installing Git.

Then run:

```bash
git --version
```

If a Git version appears, try the clone command again.

---

## Challenges and Lessons Learned

This project reinforced several practical machine learning lessons.

A model can be mathematically correct while still having limited predictive power.

Successful optimization and strong prediction are not the same thing.

Gradient Descent requires additional choices such as feature scaling, learning rate, and number of epochs.

The Normal Equation calculates the coefficients directly without iterative updates.

Another important lesson is reproducibility. An analytics project is more useful when another person can download the repository, recreate the environment, and reproduce the analysis without assistance.

Technical results should also be translated into language that non-technical stakeholders can understand.

---

## Future Improvements

Future extensions could include:

- Add smoking status
- Add region
- Compare single-feature and multi-feature regression
- Introduce train, validation, and test splits
- Compare additional learning rates
- Add early stopping
- Explore additional regression methods

---

## Reproducibility and Peer Validation

This repository is designed so that another user can reproduce the project by following only the instructions in this README.

The setup instructions support both Windows and macOS.

For the course assignment, a peer will independently test these instructions without assistance from the project author.

Any problems discovered during peer validation will be used to improve the documentation.

---

## Data Source

The project uses the insurance premium prediction dataset provided with the course instructional materials.

The dataset was originally sourced from Kaggle for educational use.

---

## Acknowledgment

This project was developed as part of MSBA 265 coursework and builds upon instructional materials provided by the course instructor.

The instructional code and concepts were extended, tested, documented, and interpreted as a reproducible portfolio project.

The purpose of this repository is to demonstrate understanding of:

- linear regression
- Normal Equation
- Gradient Descent
- model evaluation
- reproducible analytics workflows
- technical communication

---

## Author

Fangqi Hu

MS Business Analytics  
University of the Pacific
