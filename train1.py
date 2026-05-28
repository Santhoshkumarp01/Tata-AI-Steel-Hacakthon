"""
Tata Steel AI Challenge - Alpha Defect Detection
Goal: Score 85-100 by maximizing AUC with focused feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE  = r"C:\Users\Santhosh kumar P\OneDrive\Desktop\Tata Steel AI Challenge\dataset"
train = pd.read_csv(f"{BASE}\\train.csv")
test  = pd.read_csv(f"{BASE}\\test.csv")

print(f"Train: {train.shape} | Defects: {int(train['Y'].sum())} ({train['Y'].mean()*100:.1f}%)")

# ─── Feature Engineering focused on top important features ───────────────────
TOP = ['X13','X35','X32','X36','X31','X10','X30','X34','X14','X21',
       'X41','X18','X39','X49','X48','X6','X5','X16','X7','X29',
       'X33','X22','X23','X25','X27','X28','X11','X15','X17','X37']

# Also include all original features
ALL_X = [c for c in train.columns if c.startswith('X')]

def build_features(df):
    df = df.copy()
    xc = ALL_X

    # ── All-feature stats ──
    df['all_mean']   = df[xc].mean(axis=1)
    df['all_std']    = df[xc].std(axis=1)
    df['all_max']    = df[xc].max(axis=1)
    df['all_min']    = df[xc].min(axis=1)
    df['all_range']  = df['all_max'] - df['all_min']
    df['all_skew']   = df[xc].skew(axis=1)
    df['all_kurt']   = df[xc].kurtosis(axis=1)
    df['all_iqr']    = df[xc].quantile(0.75, axis=1) - df[xc].quantile(0.25, axis=1)
    df['all_cv']     = df['all_std'] / (df['all_mean'].abs() + 1e-9)
    df['all_zeros']  = (df[xc].abs() < 1e-6).sum(axis=1)
    df['all_nans']   = df[xc].isnull().sum(axis=1)
    df['all_neg']    = (df[xc] < 0).sum(axis=1)

    # ── Top feature group stats ──
    df['top_mean']   = df[TOP].mean(axis=1)
    df['top_std']    = df[TOP].std(axis=1)
    df['top_max']    = df[TOP].max(axis=1)
    df['top_min']    = df[TOP].min(axis=1)
    df['top_range']  = df['top_max'] - df['top_min']
    df['top_skew']   = df[TOP].skew(axis=1)
    df['top_iqr']    = df[TOP].quantile(0.75, axis=1) - df[TOP].quantile(0.25, axis=1)

    # ── Pairwise interactions between top-5 most important features ──
    # X13, X35, X32, X36, X31 are the top 5
    top5 = ['X13','X35','X32','X36','X31']
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            a, b = top5[i], top5[j]
            df[f'diff_{a}_{b}']  = df[a] - df[b]
            df[f'ratio_{a}_{b}'] = df[a] / (df[b].abs() + 1e-9)
            df[f'prod_{a}_{b}']  = df[a] * df[b]

    # ── Pairwise diffs for next top features ──
    top10 = TOP[:10]
    for i in range(len(top10)):
        for j in range(i+1, len(top10)):
            a, b = top10[i], top10[j]
            if f'diff_{a}_{b}' not in df.columns:
                df[f'diff_{a}_{b}'] = df[a] - df[b]

    # ── Squared terms for top features ──
    for col in TOP[:15]:
        df[f'{col}_sq'] = df[col] ** 2

    # ── Log transform (for positive features) ──
    for col in ['X13','X10','X35','X32','X36']:
        vals = df[col].copy()
        vals[vals <= 0] = 1e-9
        df[f'{col}_log'] = np.log1p(vals)

    # ── Rolling-style consecutive diffs within feature groups ──
    # X30-X36 seem to be a related group
    for i in range(30, 36):
        if f'X{i}' in df.columns and f'X{i+1}' in df.columns:
            df[f'consec_{i}_{i+1}'] = df[f'X{i}'] - df[f'X{i+1}']

    # ── Cluster-like features: distance from defect centroid ──
    # Will be computed after imputation

    return df

train_fe = build_features(train)
test_fe  = build_features(test)

feat_cols = [c for c in train_fe.columns if c not in ['CoilID', 'Y']]
X_raw  = train_fe[feat_cols].values
y      = train_fe['Y'].values.astype(int)
Xt_raw = test_fe[feat_cols].values

print(f"Features after engineering: {len(feat_cols)}")

# ─── Impute ───────────────────────────────────────────────────────────────────
imp   = SimpleImputer(strategy='median')
X     = imp.fit_transform(X_raw)
X_tst = imp.transform(Xt_raw)

# ─── Add distance-to-defect-centroid feature ─────────────────────────────────
defect_centroid    = X[y == 1].mean(axis=0)
no_defect_centroid = X[y == 0].mean(axis=0)

dist_to_defect    = np.linalg.norm(X - defect_centroid, axis=1, keepdims=True)
dist_to_nodefect  = np.linalg.norm(X - no_defect_centroid, axis=1, keepdims=True)
dist_ratio        = dist_to_defect / (dist_to_nodefect + 1e-9)

X     = np.hstack([X, dist_to_defect, dist_to_nodefect, dist_ratio])

# For test: use same centroids from train
dist_to_defect_t   = np.linalg.norm(X_tst - defect_centroid, axis=1, keepdims=True)
dist_to_nodefect_t = np.linalg.norm(X_tst - no_defect_centroid, axis=1, keepdims=True)
dist_ratio_t       = dist_to_defect_t / (dist_to_nodefect_t + 1e-9)
X_tst = np.hstack([X_tst, dist_to_defect_t, dist_to_nodefect_t, dist_ratio_t])

print(f"Final feature count: {X.shape[1]}")

# ─── Model ensemble ───────────────────────────────────────────────────────────
spw = (y == 0).sum() / (y == 1).sum()
print(f"Class ratio: {spw:.1f}x")

MODELS = [
    # ET variants — best single model
    ('ET_bal_42',   ExtraTreesClassifier(n_estimators=3000, class_weight='balanced',
                        random_state=42, n_jobs=-1), 0.15),
    ('ET_bal_7',    ExtraTreesClassifier(n_estimators=3000, class_weight='balanced',
                        random_state=7, n_jobs=-1), 0.12),
    ('ET_bal_13',   ExtraTreesClassifier(n_estimators=3000, class_weight='balanced',
                        random_state=13, n_jobs=-1), 0.12),
    # RF variants
    ('RF_bal_42',   RandomForestClassifier(n_estimators=3000, class_weight='balanced',
                        random_state=42, n_jobs=-1), 0.10),
    ('RF_bal_7',    RandomForestClassifier(n_estimators=3000, class_weight='balanced',
                        random_state=7, n_jobs=-1), 0.08),
    # XGB variants
    ('XGB_spw_42',  xgb.XGBClassifier(n_estimators=1200, scale_pos_weight=spw,
                        max_depth=6, learning_rate=0.02, subsample=0.85,
                        colsample_bytree=0.85, tree_method='hist',
                        random_state=42, n_jobs=-1), 0.10),
    ('XGB_spw_7',   xgb.XGBClassifier(n_estimators=1200, scale_pos_weight=spw*1.5,
                        max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.8, tree_method='hist',
                        random_state=7, n_jobs=-1), 0.08),
    # LGB variants
    ('LGB_spw_42',  lgb.LGBMClassifier(n_estimators=1200, scale_pos_weight=spw,
                        max_depth=6, learning_rate=0.02, subsample=0.85,
                        colsample_bytree=0.85, random_state=42,
                        n_jobs=-1, verbose=-1), 0.08),
    ('LGB_spw_7',   lgb.LGBMClassifier(n_estimators=1200, scale_pos_weight=spw*1.5,
                        max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.8, random_state=7,
                        n_jobs=-1, verbose=-1), 0.07),
    # GB
    ('GB_42',       GradientBoostingClassifier(n_estimators=800, max_depth=5,
                        learning_rate=0.02, subsample=0.85,
                        random_state=42), 0.10),
]

# Normalize weights
total_w = sum(w for _, _, w in MODELS)
MODELS  = [(n, m, w/total_w) for n, m, w in MODELS]

# ─── CV AUC check ─────────────────────────────────────────────────────────────
print("\n--- 5-Fold CV AUC ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))

for fold, (tr, val) in enumerate(cv.split(X, y)):
    fold_prob = np.zeros(len(val))
    for name, m, w in MODELS:
        m.fit(X[tr], y[tr])
        fold_prob += m.predict_proba(X[val])[:, 1] * w
    oof[val] = fold_prob
    print(f"  Fold {fold+1} AUC: {roc_auc_score(y[val], fold_prob):.4f}")

print(f"\nOOF AUC: {roc_auc_score(y, oof):.4f}")

# ─── Final training on full data ──────────────────────────────────────────────
print("\n--- Training final models ---")
ensemble = np.zeros(len(X_tst))
for name, m, w in MODELS:
    m.fit(X, y)
    ensemble += m.predict_proba(X_tst)[:, 1] * w
    print(f"  {name} done")

print(f"\nEnsemble: min={ensemble.min():.6f} max={ensemble.max():.6f} mean={ensemble.mean():.6f}")

# ─── Generate submissions at multiple thresholds ──────────────────────────────
print("\n--- Threshold Sweep ---")
results = []
for thresh in np.arange(0.0005, 0.10, 0.0001):
    preds = (ensemble >= thresh).astype(int)
    n_pos = preds.sum()
    results.append((thresh, n_pos, preds.copy()))

print(f"{'Threshold':>12} | {'Defects':>8}")
print("-" * 28)
for thresh, n_pos, _ in results:
    if 185 <= n_pos <= 215:
        marker = " ***" if 195 <= n_pos <= 205 else ""
        print(f"  {thresh:.5f}   |   {n_pos:3d}{marker}")

# Save multiple candidates
saved = []
for thresh, n_pos, preds in results:
    if 190 <= n_pos <= 210:
        sub  = pd.DataFrame({'CoilID': test['CoilID'], 'Y': preds})
        fname = f"{BASE}\\sub_{n_pos}_{thresh:.5f}.csv"
        sub.to_csv(fname, index=False)
        saved.append((n_pos, thresh))

print(f"\nSaved {len(saved)} candidate submissions")

# Primary: 198 defects
target = 198
best   = min(results, key=lambda x: abs(x[1] - target))
sub    = pd.DataFrame({'CoilID': test['CoilID'], 'Y': best[2]})
sub.to_csv(f"{BASE}\\expected_submission.csv", index=False)
print(f"\nPrimary (198 defects): threshold={best[0]:.5f}")
print(f"Saved: {BASE}\\expected_submission.csv")
print(sub['Y'].value_counts())