# ============================================================
# PART 2 & 3: MODELING AND RMSE ESTIMATION
# ============================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# LOAD CLEANED DATASETS (From Part 1)
# The training ground: 100 observations 
X_train = pd.read_csv("X_clean.csv")
# The target: 1,000 new observations
X_new = pd.read_csv("Xnew_clean.csv")
y_train = pd.read_csv("y.csv").values.ravel()

print(f"Training on {X_train.shape[0]} observations with {X_train.shape[1]} features.")

# DEFINE THE MODEL (Part 2)
# Using a Pipeline to ensure StandardScaler is applied within Cross-Validation.
# LassoCV handles the 'Curse of Dimensionality' by selecting key features
model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=10, random_state=42, max_iter=10000)
)

# FIT THE MODEL 
model.fit(X_train, y_train)

# EXTRACT DIAGNOSTICS FOR THE REPORT 
lasso_step = model.named_steps['lassocv']
active_features = np.sum(lasso_step.coef_ != 0)
print(f"Best Alpha found: {lasso_step.alpha_:.4f}")
print(f"Number of selected features: {active_features} / {X_train.shape[1]}")

# 5/ GENERATE PREDICTIONS (Part 2) 
# Predicting y_hat for the 1,000 unseen targets 
y_pred = model.predict(X_new)

# ESTIMATE RMSE (Part 3) 
# Extracting the MSE from the 10-fold cross-validation for the best alpha.
best_alpha_idx = np.where(lasso_step.alphas_ == lasso_step.alpha_)[0][0]
avg_mse = np.mean(lasso_step.mse_path_[best_alpha_idx])
# Calculate RM^SE: Square root of the mean squared error 
estimated_rmse = np.sqrt(avg_mse)

print(f"\n--- Self-Assessment ---")
print(f"Estimated RMSE (RM^SE): {estimated_rmse:.6f}")

# SAVE ARTIFACTS (Strict Formatting)
# Replace IDs if necessary; filenames must be exact
student_ids = "s253050_s253043_s225031" 
pred_filename = f"predictions_{student_ids}.csv"
rmse_filename = f"estimatedRMSE_{student_ids}.csv"

# Save 1,000 predictions (Strictly no headers) 
pd.DataFrame(y_pred).to_csv(pred_filename, index=False, header=False)

# Save single RMSE estimate (Strictly no headers) 
pd.DataFrame([estimated_rmse]).to_csv(rmse_filename, index=False, header=False)

print(f"\nFiles generated successfully:")
print(f"1. {pred_filename}")
print(f"2. {rmse_filename}")