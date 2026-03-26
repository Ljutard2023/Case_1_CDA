import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 1/ LOAD (On repart des fichiers propres de la Partie 1)
X_train = pd.read_csv("X_clean.csv")
X_new = pd.read_csv("Xnew_clean.csv")
y_train = pd.read_csv("y.csv").values.ravel()

# 2/ CONSTRUCTION DU PIPELINE ULTIME
# On ajoute une étape de sélection de variables (SelectKBest) avant le modèle
pipe = Pipeline([
    ("variance", VarianceThreshold()), # Supprime les colonnes constantes
    ("scaler", RobustScaler()),        # Plus robuste aux outliers
    ("select", SelectKBest(score_func=f_regression)), # Garde les meilleures variables
    ("svr", SVR(kernel="linear"))      # Le modèle SVR Linéaire
])

# 3/ GRILLE DE RECHERCHE
param_grid = {
    "select__k": [15, 20, 25, 30],     # On teste combien de variables garder
    "svr__C": [0.1, 1, 10, 100],       # Force de la régularisation
    "svr__epsilon": [0.1, 0.2, 0.5]    # Largeur du "tube" d'erreur
}

grid = GridSearchCV(pipe, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

# 4/ RESULTATS
best_rmse = np.sqrt(-grid.best_score_)
print(f"Best Params: {grid.best_params_}")
print(f"--- Self-Assessment ---")
print(f"New Estimated RMSE: {best_rmse:.6f}")

# 5/ GENERATION DES FICHIERS
y_pred = grid.predict(X_new)
student_ids = "s253050_s253043_s225031"
pd.DataFrame(y_pred).to_csv(f"predictions_{student_ids}.csv", index=False, header=False)
pd.DataFrame([best_rmse]).to_csv(f"estimatedRMSE_{student_ids}.csv", index=False, header=False)