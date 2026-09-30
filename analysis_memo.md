# Analysis Memo

The BMI-only baseline model produced an R² of 0.0394, meaning that BMI alone explained about 3.94% of the variation in medical expenses.

After adding age as a second feature, the R² increased to 0.1173.

This represents an improvement of about 0.0778, or 7.78 percentage points.

The Normal Equation and Gradient Descent produced the same coefficients: w0 = -6437.35, w1 = 333.39 for BMI, and w2 = 241.90 for age.

The two methods should produce very similar results because both are solving the same least-squares regression problem.

The Normal Equation reaches the solution directly, while Gradient Descent approaches the same solution through repeated updates that reduce prediction error.

The positive age coefficient of 241.90 means that, holding BMI constant, each additional year of age is associated with about $241.90 higher predicted medical expenses.

The BMI coefficient of 333.39 means that, holding age constant, a one-unit increase in BMI is associated with about $333.39 higher predicted medical expenses.

The two-feature model performs better than the BMI-only baseline, but its R² of 0.1173 is still relatively low.

This means that most of the variation in medical expenses is still not explained by BMI and age alone.

For that reason, I would not consider this model strong enough for real-world deployment or high-stakes business decisions.

A stronger model would likely need additional predictors such as smoking status, region, children, or interaction terms, followed by validation on separate data before deployment.