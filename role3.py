import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# =========================================================
# 0. Config
# =========================================================
TRAIN_HH = "train_val_households.csv"
TRAIN_PERS = "train_val_persons.csv"
TRAIN_TRIPS = "train_val_trips.csv"

TEST_HH = "test_households.csv"
TEST_PERS = "test_persons.csv"
TEST_TRIPS = "test_trips.csv"

N_SPLITS = 5
RANDOM_STATE = 42

TARGET_COL = "mode"
GROUP_COL = "hhid"
ID_COL = "tripid"
PERSON_COL = "persid"


# =========================================================
# 1. Load and Merge
# =========================================================
def load_and_merge():
    print("Loading and merging data...")

    hh_train = pd.read_csv(TRAIN_HH)
    pers_train = pd.read_csv(TRAIN_PERS)
    trips_train = pd.read_csv(TRAIN_TRIPS)

    hh_test = pd.read_csv(TEST_HH)
    pers_test = pd.read_csv(TEST_PERS)
    trips_test = pd.read_csv(TEST_TRIPS)

    train = (
        trips_train
        .merge(pers_train, on=[GROUP_COL, PERSON_COL], how="left")
        .merge(hh_train, on=GROUP_COL, how="left")
    )

    test = (
        trips_test
        .merge(pers_test, on=[GROUP_COL, PERSON_COL], how="left")
        .merge(hh_test, on=GROUP_COL, how="left")
    )

    print("Train merged shape:", train.shape)
    print("Test merged shape :", test.shape)

    return train, test


# =========================================================
# 2. Missing Value Handling
#    Use the training-set statistic to fill the train/test sets
# =========================================================
def fill_missing(train, test):
    print("Handling missing values...")

    train = train.copy()
    test = test.copy()

    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [ID_COL, GROUP_COL, PERSON_COL]]

    for col in cat_cols:
        if col in train.columns:
            train[col] = train[col].fillna("Missing")
        if col in test.columns:
            test[col] = test[col].fillna("Missing")

    for col in num_cols:
        median_value = train[col].median()
        if col in train.columns:
            train[col] = train[col].fillna(median_value)
        if col in test.columns:
            test[col] = test[col].fillna(median_value)

    return train, test


# =========================================================
# 3. Logical Quality Check
# =========================================================
def logical_quality_check(df):
    df = df.copy()

    if "anywork" in df.columns and "anywfh" in df.columns:
        df["flag_wfh_anomaly"] = (
            (df["anywork"] == "No") & (df["anywfh"] == "Yes")
        ).astype(int)

    if "cumdist" in df.columns and "travtime" in df.columns:
        df["cumdist"] = pd.to_numeric(df["cumdist"], errors="coerce")
        df["travtime"] = pd.to_numeric(df["travtime"], errors="coerce")
        safe_travtime = df["travtime"].clip(lower=1)
        df["speed_kmh"] = df["cumdist"] / (safe_travtime / 60.0)
        df["flag_speed_anomaly"] = (df["speed_kmh"] > 120).astype(int)

    return df


# =========================================================
# 4. Role 2 static feature engineering
# =========================================================
def extract_age_median(age_str):
    if pd.isna(age_str) or age_str == "Missing":
        return np.nan
    nums = re.findall(r"\d+", str(age_str))
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    return np.nan


def static_feature_engineering(df):
    df = df.copy()

    # WFH aggregation
    wfh_cols = ["wfhmon", "wfhtue", "wfhwed", "wfhthu", "wfhfri", "wfhsat", "wfhsun"]
    if all(col in df.columns for col in wfh_cols):
        df["wfh_days_per_week"] = (df[wfh_cols] == "Yes").sum(axis=1)
        df.drop(columns=wfh_cols, inplace=True)

    # Age group -> numeric
    age_cols = ["youngestgroup_5", "aveagegroup_5", "oldestgroup_5"]
    if all(col in df.columns for col in age_cols):
        df["youngest_age_num"] = df["youngestgroup_5"].apply(extract_age_median)
        df["avg_age_num"] = df["aveagegroup_5"].apply(extract_age_median)
        df["oldest_age_num"] = df["oldestgroup_5"].apply(extract_age_median)

        df["has_kids"] = (df["youngest_age_num"] < 15).astype(int)
        df["has_seniors"] = (df["oldest_age_num"] >= 65).astype(int)
        df.drop(columns=age_cols, inplace=True)

    # Mobility cross feature
    if "carlicence" in df.columns and "totalvehs" in df.columns:
        has_licence = df["carlicence"].isin(
            ["Full Licence", "Probationary Licence"]
        ).astype(int)
        df["veh_per_driver"] = df["totalvehs"] / (has_licence + 0.001)

    cols_to_drop = ["activities", "starthour", "arrhour", "travtime", "homeregion_ASGS"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df


# =========================================================
# 5. Fold-safe time_of_day
#    Only used the information of train fold to calculate the boundary of binning
# =========================================================
def compute_time_bin_edges(train_fold):
    # Morning peak：quantiles of public transport departure times in the morning
    pub_am_mask = (
        (train_fold[TARGET_COL] == "PUBLICTRANSPORT") &
        (train_fold["startime"] < 720)
    )
    pub_am_times = train_fold.loc[pub_am_mask, "startime"]
    am_peak_start = pub_am_times.quantile(0.15) if not pub_am_times.empty else 390
    am_peak_end = pub_am_times.quantile(0.85) if not pub_am_times.empty else 660

    # Evening peak：quantiles of public transport departure times in the afternoon
    pub_pm_mask = (
        (train_fold[TARGET_COL] == "PUBLICTRANSPORT") &
        (train_fold["startime"] >= 720)
    )
    pub_pm_times = train_fold.loc[pub_pm_mask, "startime"]
    pm_peak_start = pub_pm_times.quantile(0.15) if not pub_pm_times.empty else 780
    pm_peak_end = pub_pm_times.quantile(0.85) if not pub_pm_times.empty else 1080

    # School drop-off and pick-up peak hours：quantiles of afternoon departure times for Education-related trips.
    if "trippurp" in train_fold.columns:
        edu_pm_mask = (
            (train_fold["trippurp"] == "Education") &
            (train_fold["startime"] >= 720)
        )
        edu_pm_times = train_fold.loc[edu_pm_mask, "startime"]
        school_peak_start = edu_pm_times.quantile(0.10) if not edu_pm_times.empty else 900
        school_peak_end = edu_pm_times.quantile(0.90) if not edu_pm_times.empty else 960
    else:
        school_peak_start, school_peak_end = 900, 960

    return {
        "am_peak_start": am_peak_start,
        "am_peak_end": am_peak_end,
        "pm_peak_start": pm_peak_start,
        "pm_peak_end": pm_peak_end,
        "school_peak_start": school_peak_start,
        "school_peak_end": school_peak_end,
    }


def add_time_of_day(df, edges):
    df = df.copy()

    if "startime" not in df.columns or "dayType" not in df.columns:
        return df

    def assign_datadriven_time(row):
        mins = row["startime"]
        day_type = row["dayType"]

        if day_type == "Weekend":
            if mins < 540:
                return "Weekend_Morning"
            elif mins < 1080:
                return "Weekend_Daytime"
            else:
                return "Weekend_Evening"
        else:
            if edges["am_peak_start"] <= mins <= edges["am_peak_end"]:
                return "Morning_Commute_Peak"
            elif edges["school_peak_start"] <= mins <= edges["school_peak_end"]:
                return "School_Pickup_Peak"
            elif edges["pm_peak_start"] <= mins <= edges["pm_peak_end"]:
                return "Evening_Commute_Peak"
            elif mins < edges["am_peak_start"]:
                return "Night_Early_Morning"
            elif edges["am_peak_end"] < mins < edges["school_peak_start"]:
                return "Midday_OffPeak"
            else:
                return "Late_Evening"

    df["time_of_day"] = df.apply(assign_datadriven_time, axis=1)
    return df


# =========================================================
# 6. Prepare matrices
# =========================================================
def align_categories(train_x, valid_x, test_x):
    cat_cols = train_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for col in cat_cols:
        combined = pd.concat(
            [
                train_x[col].astype(str),
                valid_x[col].astype(str),
                test_x[col].astype(str)
            ],
            axis=0
        )
        categories = pd.Index(sorted(combined.unique()))

        train_x[col] = pd.Categorical(train_x[col].astype(str), categories=categories)
        valid_x[col] = pd.Categorical(valid_x[col].astype(str), categories=categories)
        test_x[col] = pd.Categorical(test_x[col].astype(str), categories=categories)

    return train_x, valid_x, test_x, cat_cols


def get_feature_columns(train_df, test_df):
    drop_cols = [TARGET_COL, ID_COL, GROUP_COL, PERSON_COL]
    feature_cols = [c for c in train_df.columns if c not in drop_cols and c in test_df.columns]
    return feature_cols


# =========================================================
# 7. CV Training
# =========================================================
def run_groupkfold_lgb(train_df, test_df):
    feature_cols = get_feature_columns(train_df, test_df)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_df[TARGET_COL].copy())
    groups = train_df[GROUP_COL].copy()

    n_classes = len(label_encoder.classes_)
    oof_pred = np.zeros((len(train_df), n_classes))
    test_pred = np.zeros((len(test_df), n_classes))
    fold_scores = []

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1
    }

    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
        print(f"\n================ Fold {fold} ================")

        train_fold = train_df.iloc[tr_idx].copy()
        valid_fold = train_df.iloc[va_idx].copy()
        test_fold = test_df.copy()

        # fold-safe target-dependent bins
        edges = compute_time_bin_edges(train_fold)

        train_fold = add_time_of_day(train_fold, edges)
        valid_fold = add_time_of_day(valid_fold, edges)
        test_fold = add_time_of_day(test_fold, edges)

        X_tr = train_fold[feature_cols].copy()
        X_va = valid_fold[feature_cols].copy()
        X_te = test_fold[feature_cols].copy()

        y_tr = label_encoder.transform(train_fold[TARGET_COL])
        y_va = label_encoder.transform(valid_fold[TARGET_COL])

        X_tr, X_va, X_te, cat_cols = align_categories(X_tr, X_va, X_te)

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)
        te_pred = model.predict_proba(X_te, num_iteration=model.best_iteration_)

        oof_pred[va_idx] = va_pred
        test_pred += te_pred / N_SPLITS

        fold_logloss = log_loss(y_va, va_pred, labels=np.arange(n_classes))
        fold_scores.append(fold_logloss)

        print(f"Fold {fold} best_iteration: {model.best_iteration_}")
        print(f"Fold {fold} logloss      : {fold_logloss:.6f}")

    overall_logloss = log_loss(y_all, oof_pred, labels=np.arange(n_classes))

    print("\n================ CV Summary ================")
    print("Classes:", list(label_encoder.classes_))
    print("Fold loglosses:", [round(x, 6) for x in fold_scores])
    print(f"CV mean logloss: {np.mean(fold_scores):.6f}")
    print(f"CV std         : {np.std(fold_scores):.6f}")
    print(f"OOF logloss    : {overall_logloss:.6f}")

    return {
        "feature_cols": feature_cols,
        "label_encoder": label_encoder,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "fold_scores": fold_scores,
        "overall_logloss": overall_logloss
    }


# =========================================================
# 8. Save outputs
# =========================================================
def save_outputs(train_df, test_df, result):
    class_names = list(result["label_encoder"].classes_)

    oof_df = pd.DataFrame(result["oof_pred"], columns=class_names)
    oof_df.insert(0, ID_COL, train_df[ID_COL].values)
    oof_df.insert(1, GROUP_COL, train_df[GROUP_COL].values)
    oof_df["true_mode"] = train_df[TARGET_COL].values
    oof_df.to_csv("oof_pred_role3_lgb.csv", index=False)

    test_pred_df = pd.DataFrame(result["test_pred"], columns=class_names)
    test_pred_df.insert(0, ID_COL, test_df[ID_COL].values)
    test_pred_df.to_csv("test_pred_role3_lgb.csv", index=False)

    submission = test_pred_df.copy()
    prob_cols = class_names
    submission[prob_cols] = submission[prob_cols].div(
        submission[prob_cols].sum(axis=1),
        axis=0
    )
    submission.to_csv("submission_role3_lgb.csv", index=False)

    summary_df = pd.DataFrame({
        "fold": list(range(1, len(result["fold_scores"]) + 1)),
        "logloss": result["fold_scores"]
    })
    summary_df.loc[len(summary_df)] = ["mean", np.mean(result["fold_scores"])]
    summary_df.loc[len(summary_df)] = ["std", np.std(result["fold_scores"])]
    summary_df.loc[len(summary_df)] = ["oof", result["overall_logloss"]]
    summary_df.to_csv("cv_summary_role3_lgb.csv", index=False)

    print("\nSaved files:")
    print("- oof_pred_role3_lgb.csv")
    print("- test_pred_role3_lgb.csv")
    print("- submission_role3_lgb.csv")
    print("- cv_summary_role3_lgb.csv")


# =========================================================
# 9. Main
# =========================================================
def main():
    train, test = load_and_merge()
    train, test = fill_missing(train, test)

    train = logical_quality_check(train)
    test = logical_quality_check(test)

    # static part of role2 features
    train = static_feature_engineering(train)
    test = static_feature_engineering(test)

    result = run_groupkfold_lgb(train, test)
    save_outputs(train, test, result)

    print("\nRole 3 pipeline completed successfully.")


if __name__ == "__main__":
    main()
