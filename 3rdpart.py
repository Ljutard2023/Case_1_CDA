# ============================================================
# PART 3: RMSE ESTIMATION (Cross-Validation)
# ============================================================
import pandas as pd
import numpy as np

# Access the MSE results from your LassoCV model
# 'lasso_step' is the LassoCV object from your previous pipeline
# 'mse_path_' contains the Mean Squared Error for each fold and each alpha
mse_path = lasso_step.mse_path_

# Find the index of the best alpha used by the model
best_alpha_index = np.where(lasso_step.alphas_ == lasso_step.alpha_)[0][0]

# Calculate the average MSE across the 10 folds for that specific alpha
avg_mse = np.mean(mse_path[best_alpha_index])

# Calculate the Estimated RMSE (Square root of MSE)
estimated_rmse = np.sqrt(avg_mse)

print(f"--- Self-Assessment ---")
print(f"Estimated RMSE based on 10-fold CV: {estimated_rmse:.6f}")

# Save to the strictly formatted file 
# Filename MUST be: estimatedRMSE_Student1_Student2.csv 
student_ids = "s253050_s253043_s225031" 
rmse_filename = f"estimatedRMSE_{student_ids}.csv"

pd.DataFrame([estimated_rmse]).to_csv(rmse_filename, index=False, header=False)

print(f"\nFinal RMSE estimation file generated: {rmse_filename}")