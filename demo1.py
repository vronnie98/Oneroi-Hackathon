import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# =========================
# 1. read the data
# =========================
trip_train = pd.read_csv("train_val_trips.csv")
person_train = pd.read_csv("train_val_persons.csv")
house_train = pd.read_csv("train_val_households.csv")

trip_test = pd.read_csv("test_trips.csv")
person_test = pd.read_csv("test_persons.csv")
house_test = pd.read_csv("test_households.csv")

print("trip_train shape:", trip_train.shape)
print("person_train shape:", person_train.shape)
print("house_train shape:", house_train.shape)
print("trip_test shape:", trip_test.shape)
print("person_test shape:", person_test.shape)
print("house_test shape:", house_test.shape)

# =========================
# 2. merge the data
# =========================
train = trip_train.merge(
    person_train,
    on=["hhid", "persid"],
    how="left"
).merge(
    house_train,
    on="hhid",
    how="left"
)

test = trip_test.merge(
    person_test,
    on=["hhid", "persid"],
    how="left"
).merge(
    house_test,
    on="hhid",
    how="left"
)

print("Merged train shape:", train.shape)
print("Merged test shape:", test.shape)

# =========================
# 3. feature
# =========================
def add_features(df):
    df = df.copy()
    numeric_cols = [
        "cumdist", "travtime", "arrtime", "startime",
        "starthour", "totalvehs", "hhsize", "totalbikes", "triptime"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # average speed = distance / time
    if "cumdist" in df.columns and "travtime" in df.columns:
        df["avg_speed"] = df["cumdist"] / (df["travtime"].fillna(0) + 1)

    # total time
    if "arrtime" in df.columns and "startime" in df.columns:
        df["time_gap"] = df["arrtime"] - df["startime"]

    # 高峰期标记
    if "starthour" in df.columns:
        df["is_peak"] = df["starthour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # 家庭车辆/人数
    if "totalvehs" in df.columns and "hhsize" in df.columns:
        df["cars_per_person"] = df["totalvehs"] / (df["hhsize"].fillna(0) + 1)

    # 家庭自行车/人数
    if "totalbikes" in df.columns and "hhsize" in df.columns:
        df["bikes_per_person"] = df["totalbikes"] / (df["hhsize"].fillna(0) + 1)

    # 出行距离 / triptime
    if "cumdist" in df.columns and "triptime" in df.columns:
        df["dist_per_triptime"] = df["cumdist"] / (df["triptime"].fillna(0) + 1)

    # 早晚高峰
    if "starthour" in df.columns:
        df["is_morning_peak"] = df["starthour"].isin([7, 8, 9]).astype(int)
        df["is_evening_peak"] = df["starthour"].isin([16, 17, 18]).astype(int)

    return df

train = add_features(train)
test = add_features(test)

# =========================
# 4. tag mapping
# =========================
target_order = [
    "DRIVE",
    "PASSENGER",
    "PUBLICTRANSPORT",
    "CYCLE",
    "WALK",
    "OTHER"
]

label_map = {label: idx for idx, label in enumerate(target_order)}
inverse_label_map = {idx: label for label, idx in label_map.items()}

train["mode_label"] = train["mode"].map(label_map)

if train["mode_label"].isna().any():
    unknown_modes = train.loc[train["mode_label"].isna(), "mode"].unique()
    raise ValueError(f"发Unmapped mode tag: {unknown_modes}")

# =========================
# 5. select the feature
# =========================
drop_cols = ["mode", "mode_label", "tripid"]
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
y = train["mode_label"]

X_test_final = test.drop(columns=[c for c in ["tripid"] if c in test.columns])


missing_in_test = set(X.columns) - set(X_test_final.columns)
missing_in_train = set(X_test_final.columns) - set(X.columns)

if missing_in_test:
    print("These columns are in train but missing in test:", missing_in_test)
if missing_in_train:
    print("These columns are in test but missing in train:", missing_in_train)

# Align Columns
common_cols = [col for col in X.columns if col in X_test_final.columns]
X = X[common_cols]
X_test_final = X_test_final[common_cols]

print("Final feature count:", len(common_cols))

# =========================
# 6. Identify Category Features

# CatBoost can directly process objects/categories
# =========================
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Number of categorical features:", len(cat_features))
print("Categorical features:", cat_features)

# Handling Missing Values
# Numeric columns: Preserve NaN for missing values
# Categorical columns: Convert to string and fill in missing values
for col in cat_features:
    X[col] = X[col].astype(str).fillna("Missing")
    X_test_final[col] = X_test_final[col].astype(str).fillna("Missing")

# =========================
# 7. Divide the training set / validation set
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# =========================
# 8. train CatBoost model
# =========================
model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# =========================
# 9. evaluate the training set
# =========================
val_probs = model.predict_proba(X_val)
val_logloss = log_loss(y_val, val_probs, labels=list(range(len(target_order))))
print(f"Validation Log Loss: {val_logloss:.6f}")

# =========================
# 10. predict the probability of the training set
# =========================
test_probs = model.predict_proba(X_test_final)

# check the probability is closer to 1
row_sums = test_probs.sum(axis=1)
print("Probability row sums min/max:", row_sums.min(), row_sums.max())

# =========================
# 11. submission
# =========================
submission = pd.DataFrame({
    "tripid": test["tripid"],
    "DRIVE": test_probs[:, 0],
    "PASSENGER": test_probs[:, 1],
    "PUBLICTRANSPORT": test_probs[:, 2],
    "CYCLE": test_probs[:, 3],
    "WALK": test_probs[:, 4],
    "OTHER": test_probs[:, 5],
})

print(submission.head())

submission.to_csv("submission_catboost_baseline.csv", index=False)
print("Saved: submission_catboost_baseline.csv")
