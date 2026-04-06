# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor



"""
Predicting Distributions - pd2
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 probabilistic predictions
""" 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from distribution_regressor import DistributionRegressor

# Učitavanje stvarnih loto podataka iz CSV (bez random/sintetičkih podataka)
csv_path = "/Users/4c/Desktop/GHQ/data/loto7hh_4592_k27.csv"
df = pd.read_csv(csv_path)
cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
draws = df[cols].values.astype(float)
feature_cols = [f"f{i+1}" for i in range(7)]

# Za postojeći comparison tok koristimo jedan cilj (Num1),
# a ispod dodajemo punu predikciju svih 7 brojeva.
X = pd.DataFrame(draws[:-1], columns=feature_cols)          # prethodna kombinacija
Y = draws[1:]           # sledeća kombinacija
y = Y[:, 0]             # Num1 za poređenje modela

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=39
)

print("="*70)
print("Training Both Models")
print("="*70)

# Standard LightGBM
print("\n1. Standard LightGBM Regressor...")
lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=39,
    verbose=-1
)
lgbm_model.fit(X_train, y_train)

# DistributionRegressor
print("2. DistributionRegressor ...")
dist_model = DistributionRegressor(
    n_estimators=200,
    learning_rate=0.1,
    verbose=0,
    random_state=39
)
dist_model.fit(X_train, y_train)

print("\n" + "="*70)
print("Point Prediction Comparison")
print("="*70)

# Point predictions
lgbm_pred = lgbm_model.predict(X_test)
dist_pred = dist_model.predict(X_test)

lgbm_mse = mean_squared_error(y_test, lgbm_pred)
dist_mse = mean_squared_error(y_test, dist_pred)

lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
dist_mae = mean_absolute_error(y_test, dist_pred)

print(f"\nStandard LightGBM:")
print(f"  MSE: {lgbm_mse:.4f}")
print(f"  MAE: {lgbm_mae:.4f}")

print(f"\nDistributionRegressor:")
print(f"  MSE: {dist_mse:.4f}")
print(f"  MAE: {dist_mae:.4f}")

print("\n" + "="*70)
print("Probabilistic Evaluation")
print("="*70)

# NLL nije dostupan u DistributionRegressorCDF varijanti
print("\nDistributionRegressor NLL: N/A (nije podržano u DistributionRegressorCDF)")
print("(Standard LightGBM takođe nema NLL - nema distribucioni izlaz)")

# Prediction intervals - only available for DistributionRegressor
interval_90 = dist_model.predict_interval(X_test, confidence=0.90)
lower_90, upper_90 = interval_90[:, 0], interval_90[:, 1]
coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))

interval_50 = dist_model.predict_interval(X_test, confidence=0.50)
lower_50, upper_50 = interval_50[:, 0], interval_50[:, 1]
coverage_50 = np.mean((y_test >= lower_50) & (y_test <= upper_50))

print(f"\nPrediction Interval Coverage:")
print(f"  50% interval: {coverage_50:.1%} (target: 50%)")
print(f"  90% interval: {coverage_90:.1%} (target: 90%)")
print("(Standard LightGBM cannot provide calibrated intervals)")

print("\n" + "="*70)
print("Visualization")
print("="*70)

# Visualize predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Standard LightGBM
axes[0].scatter(y_test, lgbm_pred, alpha=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('True values')
axes[0].set_ylabel('Predicted values')
axes[0].set_title(f'Standard LightGBM\nMSE: {lgbm_mse:.4f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: DistributionRegressor point predictions
axes[1].scatter(y_test, dist_pred, alpha=0.5, s=30)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel('True values')
axes[1].set_ylabel('Predicted values')
axes[1].set_title(f'DistributionRegressor (Mean)\nMSE: {dist_mse:.4f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: DistributionRegressor with uncertainty
axes[2].errorbar(y_test, dist_pred, 
                 yerr=[dist_pred - lower_90, upper_90 - dist_pred],
                 fmt='o', alpha=0.5, markersize=4, capsize=0, linewidth=1)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[2].set_xlabel('True values')
axes[2].set_ylabel('Predicted values')
axes[2].set_title(f'DistributionRegressor with 90% Intervals\nCoverage: {coverage_90:.1%}')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/4c/Desktop/GHQ/kurzor/DistributionRegressor-main/examples/comparison_with_standard_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'comparison_with_standard_regression.png'")

print("\n" + "="*70)
print("Summary")
print("="*70)
print("""
Key Advantages of DistributionRegressor:

✓ Full probability distributions - not just point estimates
✓ Calibrated uncertainty quantification
✓ Prediction intervals with proper coverage
✓ Probabilistic evaluation (NLL)
✓ Better for decision-making under uncertainty

Point predictions (MSE/MAE) are often comparable, but DistributionRegressor
provides much richer information about prediction confidence.
""")

print("\n" + "="*70)
print("Loto 7/39 Prediction")
print("="*70)

# Predikcija sledeće kompletne kombinacije 7/39 (po pozicijama)
# pd2 = comparison: i Standard LightGBM i DistributionRegressor rezultat
X_full = draws[:-1]
Y_full = draws[1:]
X_next = pd.DataFrame(draws[-1:].astype(float), columns=feature_cols)

pred_lgbm = []
pred_dist = []
for i in range(7):
    # Standard model za poređenje
    lgbm_pos = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=39,
        verbose=-1
    )
    lgbm_pos.fit(pd.DataFrame(X_full, columns=feature_cols), Y_full[:, i])
    p_lgbm = float(lgbm_pos.predict(X_next)[0])
    pred_lgbm.append(int(np.clip(np.rint(p_lgbm), 1, 39)))

    # DistributionRegressor (mean)
    m = DistributionRegressor(
        n_estimators=200,
        learning_rate=0.1,
        verbose=0,
        random_state=39
    )
    m.fit(pd.DataFrame(X_full, columns=feature_cols), Y_full[:, i])
    p = float(m.predict(X_next)[0])
    pred_dist.append(int(np.clip(np.rint(p), 1, 39)))

pred_lgbm = np.array(pred_lgbm, dtype=int)
pred_dist = np.array(pred_dist, dtype=int)
print()
print("Predicted next loto 7/39 combination (Standard LightGBM):", pred_lgbm)
print("Predicted next loto 7/39 combination (DistributionRegressor):", pred_dist)
print()
"""
Predicted next loto 7/39 combination (Standard LightGBM): 
[ 4  9 14 19 24 32 35]

Predicted next loto 7/39 combination (DistributionRegressor): 
[ 5  9 15 20 24 30 35]
"""
