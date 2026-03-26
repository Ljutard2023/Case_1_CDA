import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# 1/ LOAD
X_train = pd.read_csv("X_clean.csv")
X_new = pd.read_csv("Xnew_clean.csv")
y_train = pd.read_csv("y.csv").values.ravel()

# 2/ PIPELINE: Scaler -> PCA -> BayesianRidge
# La PCA va condenser les 116 variables bruyantes en quelques composantes clés
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("br", BayesianRidge())
])

# 3/ GRID SEARCH
# On cherche le nombre optimal de composantes (entre 5 et 30)
param_grid = {
    "pca__n_components": [5, 10, 15, 20, 25, 30],
    "br__alpha_1": [1e-6, 1e-4],
    "br__lambda_1": [1e-6, 1e-4]
}

grid = GridSearchCV(pipe, param_grid, cv=10, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# 4/ RESULTS
best_rmse = np.sqrt(-grid.best_score_)
print(f"Best PCA Components: {grid.best_params_['pca__n_components']}")
print(f"--- Self-Assessment ---")
print(f"Estimated RMSE: {best_rmse:.6f}")

# 5/ GENERATE ARTIFACTS
y_pred = grid.predict(X_new)
student_ids = "s253050_s253043_s225031"

pd.DataFrame(y_pred).to_csv(f"predictions_{student_ids}.csv", index=False, header=False)
pd.DataFrame([best_rmse]).to_csv(f"estimatedRMSE_{student_ids}.csv", index=False, header=False)