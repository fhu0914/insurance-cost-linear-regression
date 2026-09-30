from __future__ import annotations

import csv
import math
from pathlib import Path


# ---------------------------------------------------------
# Project paths and training settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "insurance-premium-prediction"
    / "insurance.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "assignment_results.csv"

LEARNING_RATE = 0.05
EPOCHS = 10000


# ---------------------------------------------------------
# 1. Load bmi, age, and expenses
# ---------------------------------------------------------

def load_data(csv_path: Path):
    bmi = []
    age = []
    expenses = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            bmi.append(float(row["bmi"]))
            age.append(float(row["age"]))
            expenses.append(float(row["expenses"]))

    return bmi, age, expenses


# ---------------------------------------------------------
# 2. Evaluation metrics
# ---------------------------------------------------------

def mse(y_true, y_pred):
    return sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_true, y_pred)
    ) / len(y_true)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return sum(
        abs(actual - predicted)
        for actual, predicted in zip(y_true, y_pred)
    ) / len(y_true)


def r2_score(y_true, y_pred):
    y_mean = sum(y_true) / len(y_true)

    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_true, y_pred)
    )

    ss_tot = sum(
        (actual - y_mean) ** 2
        for actual in y_true
    )

    return 1 - ss_res / ss_tot


# ---------------------------------------------------------
# 3. General linear-system solver
#    Gauss-Jordan elimination
# ---------------------------------------------------------

def solve_linear_system(matrix, vector):
    n = len(vector)

    # Build augmented matrix [A | b]
    augmented = [
        [float(value) for value in matrix[i]]
        + [float(vector[i])]
        for i in range(n)
    ]

    for col in range(n):

        # Partial pivoting:
        # choose the row with the largest absolute value
        pivot_row = max(
            range(col, n),
            key=lambda r: abs(augmented[r][col]),
        )

        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix: system cannot be solved.")

        # Move pivot row into position
        augmented[col], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[col],
        )

        # Make pivot equal to 1
        pivot = augmented[col][col]

        augmented[col] = [
            value / pivot
            for value in augmented[col]
        ]

        # Eliminate this column from all other rows
        for row in range(n):
            if row == col:
                continue

            factor = augmented[row][col]

            augmented[row] = [
                augmented[row][j]
                - factor * augmented[col][j]
                for j in range(n + 1)
            ]

    return [
        augmented[i][-1]
        for i in range(n)
    ]


# ---------------------------------------------------------
# 4. Normal Equation for TWO features
#    expenses = w0 + w1*bmi + w2*age
# ---------------------------------------------------------

def fit_normal_equation_two_features(bmi, age, y):

    # Design matrix:
    # [1, bmi, age]
    X = [
        [1.0, bmi_i, age_i]
        for bmi_i, age_i in zip(bmi, age)
    ]

    columns = 3

    # Build X^T X
    xtx = [
        [
            sum(row[i] * row[j] for row in X)
            for j in range(columns)
        ]
        for i in range(columns)
    ]

    # Build X^T y
    xty = [
        sum(row[i] * target for row, target in zip(X, y))
        for i in range(columns)
    ]

    # General matrix solve
    weights = solve_linear_system(xtx, xty)

    w0, w1, w2 = weights

    return w0, w1, w2


# ---------------------------------------------------------
# 5. BMI-only baseline
# ---------------------------------------------------------

def fit_bmi_only_baseline(bmi, y):

    X = [
        [1.0, bmi_i]
        for bmi_i in bmi
    ]

    columns = 2

    xtx = [
        [
            sum(row[i] * row[j] for row in X)
            for j in range(columns)
        ]
        for i in range(columns)
    ]

    xty = [
        sum(row[i] * target for row, target in zip(X, y))
        for i in range(columns)
    ]

    w0, w1 = solve_linear_system(xtx, xty)

    return w0, w1


# ---------------------------------------------------------
# 6. Standardization
# ---------------------------------------------------------

def standardize(values):

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = math.sqrt(variance)

    standardized = [
        (value - mean) / std
        for value in values
    ]

    return standardized, mean, std


# ---------------------------------------------------------
# 7. Gradient Descent for TWO features
# ---------------------------------------------------------

def fit_gradient_descent_two_features(
    bmi,
    age,
    y,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
):

    # Standardize BOTH features
    bmi_std_values, bmi_mean, bmi_std = standardize(bmi)
    age_std_values, age_mean, age_std = standardize(age)

    # Weights in standardized feature space
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0

    n = len(y)

    for epoch in range(1, epochs + 1):

        predictions = [
            w0
            + w1 * bmi_i
            + w2 * age_i
            for bmi_i, age_i
            in zip(bmi_std_values, age_std_values)
        ]

        errors = [
            pred - actual
            for pred, actual
            in zip(predictions, y)
        ]

        grad_w0 = (
            2 / n
        ) * sum(errors)

        grad_w1 = (
            2 / n
        ) * sum(
            error * bmi_i
            for error, bmi_i
            in zip(errors, bmi_std_values)
        )

        grad_w2 = (
            2 / n
        ) * sum(
            error * age_i
            for error, age_i
            in zip(errors, age_std_values)
        )

        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        if epoch == 1 or epoch % 1000 == 0:
            current_mse = mse(y, predictions)

            print(
                f"epoch={epoch:5d} "
                f"mse={current_mse:.2f}"
            )

    # Convert standardized weights back
    # to ORIGINAL feature units

    w1_original = w1 / bmi_std
    w2_original = w2 / age_std

    w0_original = (
        w0
        - w1 * bmi_mean / bmi_std
        - w2 * age_mean / age_std
    )

    return (
        w0_original,
        w1_original,
        w2_original,
    )


# ---------------------------------------------------------
# 8. Prediction functions
# ---------------------------------------------------------

def predict_two_features(
    bmi,
    age,
    w0,
    w1,
    w2,
):

    return [
        w0
        + w1 * bmi_i
        + w2 * age_i
        for bmi_i, age_i
        in zip(bmi, age)
    ]


def predict_bmi_only(
    bmi,
    w0,
    w1,
):

    return [
        w0 + w1 * bmi_i
        for bmi_i in bmi
    ]


# ---------------------------------------------------------
# 9. Build result row
# ---------------------------------------------------------

def build_result_row(
    model_name,
    w0,
    w1,
    w2,
    y,
    predictions,
):

    return {
        "model": model_name,
        "w0_intercept": w0,
        "w1_bmi": w1,
        "w2_age": w2,
        "MSE": mse(y, predictions),
        "RMSE": rmse(y, predictions),
        "MAE": mae(y, predictions),
        "R2": r2_score(y, predictions),
    }


# ---------------------------------------------------------
# 10. Print comparison table
# ---------------------------------------------------------

def print_comparison_table(results):

    print("\nMODEL COMPARISON")
    print("-" * 105)

    header = (
        f"{'Model':<28}"
        f"{'w0':>12}"
        f"{'w1 BMI':>12}"
        f"{'w2 Age':>12}"
        f"{'MSE':>16}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'R^2':>10}"
    )

    print(header)
    print("-" * 105)

    for row in results:

        w2_display = (
            "-"
            if row["w2_age"] is None
            else f"{row['w2_age']:.2f}"
        )

        print(
            f"{row['model']:<28}"
            f"{row['w0_intercept']:>12.2f}"
            f"{row['w1_bmi']:>12.2f}"
            f"{w2_display:>12}"
            f"{row['MSE']:>16.2f}"
            f"{row['RMSE']:>12.2f}"
            f"{row['MAE']:>12.2f}"
            f"{row['R2']:>10.4f}"
        )


# ---------------------------------------------------------
# 11. Export CSV
# ---------------------------------------------------------

def export_results(results, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "w0_intercept",
        "w1_bmi",
        "w2_age",
        "MSE",
        "RMSE",
        "MAE",
        "R2",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------
# 12. Main workflow
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("TWO-FEATURE INSURANCE LINEAR REGRESSION")
    print("=" * 70)

    print("\nModel:")
    print(
        "expenses = w0 + w1*bmi + w2*age"
    )

    # Load data
    bmi, age, expenses = load_data(CSV_PATH)

    print(
        f"\nLoaded {len(expenses)} rows "
        f"from {CSV_PATH}"
    )

    # -----------------------------------------------------
    # BMI-only baseline
    # -----------------------------------------------------

    baseline_w0, baseline_w1 = (
        fit_bmi_only_baseline(
            bmi,
            expenses,
        )
    )

    baseline_predictions = predict_bmi_only(
        bmi,
        baseline_w0,
        baseline_w1,
    )

    baseline_result = build_result_row(
        "BMI-only baseline",
        baseline_w0,
        baseline_w1,
        None,
        expenses,
        baseline_predictions,
    )

    # -----------------------------------------------------
    # Normal Equation
    # -----------------------------------------------------

    ne_w0, ne_w1, ne_w2 = (
        fit_normal_equation_two_features(
            bmi,
            age,
            expenses,
        )
    )

    ne_predictions = predict_two_features(
        bmi,
        age,
        ne_w0,
        ne_w1,
        ne_w2,
    )

    ne_result = build_result_row(
        "Normal Equation",
        ne_w0,
        ne_w1,
        ne_w2,
        expenses,
        ne_predictions,
    )

    # -----------------------------------------------------
    # Gradient Descent
    # -----------------------------------------------------

    print(
        "\nTraining Gradient Descent "
        "(梯度下降)..."
    )

    gd_w0, gd_w1, gd_w2 = (
        fit_gradient_descent_two_features(
            bmi,
            age,
            expenses,
        )
    )

    gd_predictions = predict_two_features(
        bmi,
        age,
        gd_w0,
        gd_w1,
        gd_w2,
    )

    gd_result = build_result_row(
        "Gradient Descent",
        gd_w0,
        gd_w1,
        gd_w2,
        expenses,
        gd_predictions,
    )

    results = [
        baseline_result,
        ne_result,
        gd_result,
    ]

    # Print table
    print_comparison_table(results)

    # -----------------------------------------------------
    # Improvement over baseline
    # -----------------------------------------------------

    improvement = (
        ne_result["R2"]
        - baseline_result["R2"]
    )

    print("\nR^2 IMPROVEMENT")
    print(
        f"BMI-only baseline R^2: "
        f"{baseline_result['R2']:.4f}"
    )

    print(
        f"BMI + age R^2:         "
        f"{ne_result['R2']:.4f}"
    )

    print(
        f"Improvement:            "
        f"{improvement:.4f} "
        f"({improvement * 100:.2f} percentage points)"
    )

    # -----------------------------------------------------
    # Weight comparison
    # -----------------------------------------------------

    print("\nWEIGHT COMPARISON")

    print(
        "Normal Equation:  "
        f"w0={ne_w0:.2f}, "
        f"w1={ne_w1:.2f}, "
        f"w2={ne_w2:.2f}"
    )

    print(
        "Gradient Descent: "
        f"w0={gd_w0:.2f}, "
        f"w1={gd_w1:.2f}, "
        f"w2={gd_w2:.2f}"
    )

    # Export CSV
    export_results(
        results,
        REPORT_PATH,
    )

    print(
        f"\nResults saved to:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()