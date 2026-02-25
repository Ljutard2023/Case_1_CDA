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

# 2/ Create a Pipeline: Scaling + Lasso with Cross-Validation
# Since Lasso penalizes coefficients, features must be on the same scale.
# We use LassoCV to automatically find the optimal alpha (regularization strength).
model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=10, random_state=42, max_iter=10000)
)

# 3/ Fit the model to extract the signal from the noise [cite: 4, 6]
model.fit(X_train, y_train)

# 4/ Extract model diagnostics for your report [cite: 18, 20]
lasso_step = model.named_steps['lassocv']
active_features = np.sum(lasso_step.coef_ != 0)
print(f"Best Alpha found: {lasso_step.alpha_:.4f}")
print(f"Number of selected features: {active_features} / {X_train.shape[1]}")

# 5/ Generate predictions for the 1,000 unseen targets [cite: 4, 11]
y_pred = model.predict(X_new)

# 6/ Save the predictions strictly following the rules 
# Note: Filenames must be EXACT: predictions_Student1_Student2.csv
# Note: Header must be set to False to meet submission requirements.
student_ids = "YourID1_YourID2_YourID3"  # <--- UPDATE THIS
output_name = f"predictions_{student_ids}.csv"

pd.DataFrame(y_pred).to_csv(output_name, index=False, header=False)

print(f"\nFinal prediction file generated: {output_name}")