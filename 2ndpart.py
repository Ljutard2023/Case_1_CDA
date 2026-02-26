# ============================================================
# PART 2: PREDICTIVE MODELING (LASSO)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1/ Load cleaned datasets from Part 1
X_train = pd.read_csv("X_clean.csv")
X_new = pd.read_csv("Xnew_clean.csv")
y_train = pd.read_csv("y.csv").values.ravel()

print(f"Training on {X_train.shape[0]} observations with {X_train.shape[1]} features.")

# Create a Pipeline: Scaling + Lasso with Cross-Validation
# Since Lasso penalizes coefficients, features must be on the same scale.
# We use LassoCV to automatically find the optimal alpha (regularization strength).
model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=10, random_state=42, max_iter=10000)
)

# Fit the model to extract the signal from the noise
model.fit(X_train, y_train)

# Extract model diagnostics for report
lasso_step = model.named_steps['lassocv']
active_features = np.sum(lasso_step.coef_ != 0)
print(f"Best Alpha found: {lasso_step.alpha_:.4f}")
print(f"Number of selected features: {active_features} / {X_train.shape[1]}")

# Generate predictions for the 1,000 unseen targets
y_pred = model.predict(X_new)

# Save the predictions strictly following the rules 
# Note: Filenames must be : predictions_Student1_Student2.csv
# Note: Header must be set to False to meet submission requirements
student_ids = "s253050_s253043_s225031" 
output_name = f"predictions_{student_ids}.csv"

pd.DataFrame(y_pred).to_csv(output_name, index=False, header=False)

print(f"\nFinal prediction file generated: {output_name}")