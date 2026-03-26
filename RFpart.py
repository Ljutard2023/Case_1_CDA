import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# 1/ LOAD DATA
X_train = pd.read_csv("X_clean.csv")
X_new = pd.read_csv("Xnew_clean.csv")
y_train = pd.read_csv("y.csv").values.ravel()

print(f"Testing Random Forest on {X_train.shape[0]} samples...")

# 2/ GRID SEARCH (Rigorous Cross-Validation) [cite: 28, 43]
# We limit max_depth and min_samples_leaf to prevent overfitting on n=100
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5], 
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt', 'log2'] # Important for p=100
}

rf = RandomForestRegressor(random_state=42)
# Using 10-fold CV as required by the 'Rules of Engagement' [cite: 28, 43]
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=10, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 3/ FIT & BEST PARAMS
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

# 4/ ESTIMATE RMSE (Part 3) [cite: 25, 27, 28]
# We take the square root of the best cross-validation MSE
best_cv_mse = -grid_search.best_score_
estimated_rmse = np.sqrt(best_cv_mse)

print(f"\n--- Self-Assessment ---")
print(f"Estimated RMSE (RM^SE) via RF-CV: {estimated_rmse:.6f}")

# 5/ GENERATE PREDICTIONS (Part 2) [cite: 31]
y_pred = best_rf.predict(X_new)

# 6/ SAVE ARTIFACTS (Strict Formatting) [cite: 31, 32, 35]
student_ids = "s253050_s253043_s225031"
pred_filename = f"predictions_{student_ids}.csv"
rmse_filename = f"estimatedRMSE_{student_ids}.csv"

# No headers allowed 
pd.DataFrame(y_pred).to_csv(pred_filename, index=False, header=False)
pd.DataFrame([estimated_rmse]).to_csv(rmse_filename, index=False, header=False)

print(f"\nFiles generated: {pred_filename} and {rmse_filename}")