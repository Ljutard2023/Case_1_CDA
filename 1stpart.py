# ============================================================
# 0/ LIBRARIES
# ============================================================
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 1/ LOAD DATA
# ============================================================
train_path = "case1Data.csv"
xnew_path = "case1Data_Xnew.csv"

train_df = pd.read_csv(train_path)
xnew_df = pd.read_csv(xnew_path)

print("Train shape:", train_df.shape)
print("Xnew shape:", xnew_df.shape)
print("\nTrain head:\n", train_df.head())
print("\nXnew head:\n", xnew_df.head())


# ============================================================
# 2/ SEPARATE TARGET AND FEATURES
# ============================================================
if "y" not in train_df.columns:
    raise ValueError("Target column 'y' not found in training data.")

y = train_df["y"].copy()
X = train_df.drop(columns=["y"]).copy()

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3/ IDENTIFY VARIABLE TYPES (force C_01..C_05 as categorical)
# ============================================================
forced_cat = ["C_01", "C_02", "C_03", "C_04", "C_05"]

missing_in_X = [c for c in forced_cat if c not in X.columns]
missing_in_xnew = [c for c in forced_cat if c not in xnew_df.columns]
if missing_in_X:
    raise ValueError(f"Missing expected columns in training features X: {missing_in_X}")
if missing_in_xnew:
    raise ValueError(f"Missing expected columns in Xnew: {missing_in_xnew}")

# Convert categorical code columns to object strings (robust) and keep missing as np.nan
for c in forced_cat:
    X[c] = X[c].astype("string").replace({pd.NA: np.nan}).astype("object")
    xnew_df[c] = xnew_df[c].astype("string").replace({pd.NA: np.nan}).astype("object")

cat_cols = forced_cat[:]
num_cols = [c for c in X.columns if c not in cat_cols]

print("\nNumerical columns:", len(num_cols))
print("Categorical columns:", len(cat_cols), cat_cols)


# ============================================================
# 4/ DEFINE PREPROCESSING
# ============================================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    remainder="drop"
)


# ============================================================
# 5/ FIT ON TRAIN, TRANSFORM BOTH
# ============================================================
# Fix: ensure no pd.NA remains anywhere (scikit-learn expects np.nan)
X = X.astype("object").where(pd.notna(X), np.nan)
xnew_df = xnew_df.astype("object").where(pd.notna(xnew_df), np.nan)

# Ensure forced categorical columns stay as object
for c in forced_cat:
    X[c] = X[c].astype("object")
    xnew_df[c] = xnew_df[c].astype("object")

X_clean = preprocessor.fit_transform(X)
Xnew_clean = preprocessor.transform(xnew_df)

print("\nProcessed train shape:", X_clean.shape)
print("Processed Xnew shape:", Xnew_clean.shape)


# ============================================================
# 6/ CONVERT TO DATAFRAME
# ============================================================
feature_names = []
feature_names.extend(num_cols)

ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

X_clean_df = pd.DataFrame(
    X_clean.toarray() if hasattr(X_clean, "toarray") else X_clean,
    columns=feature_names
)
Xnew_clean_df = pd.DataFrame(
    Xnew_clean.toarray() if hasattr(Xnew_clean, "toarray") else Xnew_clean,
    columns=feature_names
)

print("\nCleaned train head:\n", X_clean_df.head())


# ============================================================
# 7/ SAVE CLEAN DATA
# ============================================================
X_clean_df.to_csv("X_clean.csv", index=False)
Xnew_clean_df.to_csv("Xnew_clean.csv", index=False)
pd.DataFrame({"y": y}).to_csv("y.csv", index=False)

print("\nSaved: X_clean.csv, Xnew_clean.csv, y.csv")
