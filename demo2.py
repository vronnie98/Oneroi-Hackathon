import os
import re
import warnings
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
    train_files = [TRAIN_HH, TRAIN_PERS, TRAIN_TRIPS]
    test_files = [TEST_HH, TEST_PERS, TEST_TRIPS]

    has_train = all(os.path.exists(f) for f in train_files)
    has_test = all(os.path.exists(f) for f in test_files)

    if not has_train and not has_test:
        raise FileNotFoundError("Error: Missing both train and test files.")

    train, test = None, None

    if has_train:
        train = pd.read_csv(TRAIN_TRIPS) \
            .merge(pd.read_csv(TRAIN_PERS), on=[PERSON_COL, GROUP_COL], how="left") \
            .merge(pd.read_csv(TRAIN_HH), on=GROUP_COL, how="left", suffixes=('', '_hh'))
        if 'travdow_hh' in train.columns:
            train = train.drop(columns=['travdow_hh'])

    if has_test:
        test = pd.read_csv(TEST_TRIPS) \
            .merge(pd.read_csv(TEST_PERS), on=[PERSON_COL, GROUP_COL], how="left") \
            .merge(pd.read_csv(TEST_HH), on=GROUP_COL, how="left", suffixes=('', '_hh'))
        if 'travdow_hh' in test.columns:
            test = test.drop(columns=['travdow_hh'])

    return train, test, has_train, has_test


# =========================================================
# 2. Missing Value Handling
# =========================================================
def handle_missing_values(train, test, has_train, has_test):
    print("Handling missing values...")
    ref_df = train if has_train else test

    cat_cols = ref_df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    num_cols = ref_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [ID_COL, GROUP_COL, PERSON_COL]]

    for col in cat_cols:
        if has_train and col in train.columns: train[col] = train[col].fillna("Missing")
        if has_test and col in test.columns:   test[col] = test[col].fillna("Missing")

    for col in num_cols:
        if has_train and col in train.columns:
            median_val = train[col].median()
        else:
            median_val = test[col].median()

        if has_train and col in train.columns: train[col] = train[col].fillna(median_val)
        if has_test and col in test.columns:   test[col] = test[col].fillna(median_val)

    return train, test


# =========================================================
# 3. Logical Quality Check
# =========================================================
def logical_quality_check(df):
    df_checked = df.copy()
    if 'anywork' in df_checked.columns and 'anywfh' in df_checked.columns:
        df_checked['flag_wfh_anomaly'] = ((df_checked['anywork'] == 'No') & (df_checked['anywfh'] == 'Yes')).astype(int)

    if 'cumdist' in df_checked.columns and 'travtime' in df_checked.columns:
        safe_travtime = df_checked['travtime'].clip(lower=1)
        df_checked['speed_kmh'] = df_checked['cumdist'] / (safe_travtime / 60.0)
        df_checked['flag_speed_anomaly'] = (df_checked['speed_kmh'] > 120).astype(int)
    return df_checked


# =========================================================
# 4. Static Feature Engineering (Upgraded v2.0)
# =========================================================
def extract_age_median(age_str):
    if pd.isna(age_str) or age_str == "Missing": return np.nan
    nums = re.findall(r'\d+', str(age_str))
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    return np.nan


def static_feature_engineering(df):
    df_feat = df.copy()

    # 1. WFH aggregation
    wfh_cols = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']
    if all(col in df_feat.columns for col in wfh_cols):
        df_feat['wfh_days_per_week'] = (df_feat[wfh_cols] == 'Yes').sum(axis=1)
        df_feat.drop(columns=wfh_cols, inplace=True)

    # 2. Age parsing
    age_cols = ['youngestgroup_5', 'aveagegroup_5', 'oldestgroup_5']
    if all(col in df_feat.columns for col in age_cols):
        df_feat['youngest_age_num'] = df_feat['youngestgroup_5'].apply(extract_age_median)
        df_feat['avg_age_num'] = df_feat['aveagegroup_5'].apply(extract_age_median)
        df_feat['oldest_age_num'] = df_feat['oldestgroup_5'].apply(extract_age_median)
        df_feat['has_kids'] = (df_feat['youngest_age_num'] < 15).astype(int)
        df_feat['has_seniors'] = (df_feat['oldest_age_num'] >= 65).astype(int)
        df_feat.drop(columns=age_cols, inplace=True)

    # 3. Mobility
    if 'carlicence' in df_feat.columns and 'totalvehs' in df_feat.columns:
        has_licence = df_feat['carlicence'].isin(['Full Licence', 'Probationary Licence']).astype(int)
        df_feat['veh_per_driver'] = df_feat['totalvehs'] / (has_licence + 0.001)

    # 4. [NEW] Household Companion Travel Feature
    # Identifies trips from the same household starting at the same time
    if 'hhid' in df_feat.columns and 'startime' in df_feat.columns and 'tripid' in df_feat.columns:
        # Count trips for the same household at the exact same start time
        df_feat['hh_trip_count_same_time'] = df_feat.groupby(['hhid', 'startime'])['tripid'].transform('count')
        # Subtracted 1 to represent companions (0 = traveling alone)
        df_feat['companion_count'] = df_feat['hh_trip_count_same_time'] - 1
        df_feat['has_hh_companion'] = (df_feat['companion_count'] > 0).astype(int)
        df_feat.drop(columns=['hh_trip_count_same_time'], inplace=True)

    # 5. [MODIFIED] Drop strictly useless columns, KEEP continuous time features
    # 'travtime', 'starthour', 'arrhour' are now preserved!
    cols_to_drop = ['activities', 'homeregion_ASGS']
    df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns], inplace=True)

    return df_feat


# =========================================================
# 5. Dynamic Feature Engineering (Fold-Safe)
# =========================================================
def compute_time_bin_edges(train_fold):
    pub_am_mask = (train_fold.get(TARGET_COL, pd.Series(dtype=str)) == 'PUBLICTRANSPORT') & (
                train_fold['startime'] < 720)
    pub_am_times = train_fold.loc[pub_am_mask, 'startime'] if not pub_am_mask.empty else pd.Series(dtype=float)
    am_peak_start = pub_am_times.quantile(0.15) if not pub_am_times.empty else 390
    am_peak_end = pub_am_times.quantile(0.85) if not pub_am_times.empty else 660

    pub_pm_mask = (train_fold.get(TARGET_COL, pd.Series(dtype=str)) == 'PUBLICTRANSPORT') & (
                train_fold['startime'] >= 720)
    pub_pm_times = train_fold.loc[pub_pm_mask, 'startime'] if not pub_pm_mask.empty else pd.Series(dtype=float)
    pm_peak_start = pub_pm_times.quantile(0.15) if not pub_pm_times.empty else 780
    pm_peak_end = pub_pm_times.quantile(0.85) if not pub_pm_times.empty else 1080

    edu_pm_mask = (train_fold.get('trippurp', pd.Series(dtype=str)) == 'Education') & (train_fold['startime'] >= 720)
    if 'trippurp' in train_fold.columns:
        edu_pm_times = train_fold.loc[edu_pm_mask, 'startime']
        school_peak_start = edu_pm_times.quantile(0.10) if not edu_pm_times.empty else 900
        school_peak_end = edu_pm_times.quantile(0.90) if not edu_pm_times.empty else 960
    else:
        school_peak_start, school_peak_end = 900, 960

    return {
        "am_peak_start": am_peak_start, "am_peak_end": am_peak_end,
        "pm_peak_start": pm_peak_start, "pm_peak_end": pm_peak_end,
        "school_peak_start": school_peak_start, "school_peak_end": school_peak_end
    }


def add_time_of_day(df, edges):
    df_feat = df.copy()
    if 'startime' not in df_feat.columns or 'dayType' not in df_feat.columns:
        return df_feat

    def assign_datadriven_time(row):
        mins = row['startime']
        day_type = row['dayType']
        if day_type == 'Weekend':
            if mins < 540:
                return 'Weekend_Morning'
            elif mins < 1080:
                return 'Weekend_Daytime'
            else:
                return 'Weekend_Evening'
        else:
            if edges["am_peak_start"] <= mins <= edges["am_peak_end"]:
                return 'Morning_Commute_Peak'
            elif edges["school_peak_start"] <= mins <= edges["school_peak_end"]:
                return 'School_Pickup_Peak'
            elif edges["pm_peak_start"] <= mins <= edges["pm_peak_end"]:
                return 'Evening_Commute_Peak'
            elif mins < edges["am_peak_start"]:
                return 'Night_Early_Morning'
            elif edges["am_peak_end"] < mins < edges["school_peak_start"]:
                return 'Midday_OffPeak'
            else:
                return 'Late_Evening'

    df_feat['time_of_day'] = df_feat.apply(assign_datadriven_time, axis=1)
    return df_feat


# =========================================================
# 6. CV Tools
# =========================================================
def align_categories(train_x, valid_x, test_x):
    cat_cols = train_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        test_series = test_x[col] if test_x is not None else pd.Series(dtype=str)
        combined = pd.concat([train_x[col].astype(str), valid_x[col].astype(str), test_series.astype(str)], axis=0)
        categories = pd.Index(sorted(combined.unique()))

        train_x[col] = pd.Categorical(train_x[col].astype(str), categories=categories)
        valid_x[col] = pd.Categorical(valid_x[col].astype(str), categories=categories)
        if test_x is not None:
            test_x[col] = pd.Categorical(test_x[col].astype(str), categories=categories)

    return train_x, valid_x, test_x, cat_cols


def get_feature_columns(train_df, test_df):
    drop_cols = [TARGET_COL, ID_COL, GROUP_COL, PERSON_COL]
    if test_df is not None:
        return [c for c in train_df.columns if c not in drop_cols and c in test_df.columns]
    return [c for c in train_df.columns if c not in drop_cols]


# =========================================================
# 7. CV Training & Fold-Safe Processing
# =========================================================
def run_groupkfold_lgb(train_df, test_df):
    print("Starting Fold-Safe CV Training...")
    feature_cols = get_feature_columns(train_df, test_df)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(train_df[TARGET_COL].copy())
    groups = train_df[GROUP_COL].copy()

    n_classes = len(label_encoder.classes_)
    oof_pred = np.zeros((len(train_df), n_classes))
    test_pred = np.zeros((len(test_df), n_classes)) if test_df is not None else None
    fold_scores = []

    params = {
        "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
        "n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31,
        "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 2.0, "random_state": RANDOM_STATE,
        "verbosity": -1, "n_jobs": -1
    }

    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, groups), start=1):
        train_fold = train_df.iloc[tr_idx].copy()
        valid_fold = train_df.iloc[va_idx].copy()
        test_fold = test_df.copy() if test_df is not None else None

        edges = compute_time_bin_edges(train_fold)
        train_fold = add_time_of_day(train_fold, edges)
        valid_fold = add_time_of_day(valid_fold, edges)
        if test_fold is not None:
            test_fold = add_time_of_day(test_fold, edges)

        current_features = feature_cols + ['time_of_day'] if 'time_of_day' in train_fold.columns else feature_cols

        X_tr = train_fold[current_features].copy()
        X_va = valid_fold[current_features].copy()
        X_te = test_fold[current_features].copy() if test_fold is not None else None

        y_tr = label_encoder.transform(train_fold[TARGET_COL])
        y_va = label_encoder.transform(valid_fold[TARGET_COL])

        X_tr, X_va, X_te, cat_cols = align_categories(X_tr, X_va, X_te)

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss",
            categorical_feature=cat_cols, callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)
        oof_pred[va_idx] = va_pred
        fold_logloss = log_loss(y_va, va_pred, labels=np.arange(n_classes))
        fold_scores.append(fold_logloss)

        print(f"Fold {fold} logloss: {fold_logloss:.6f}")

        if test_fold is not None:
            test_pred += model.predict_proba(X_te, num_iteration=model.best_iteration_) / N_SPLITS

    overall_logloss = log_loss(y_all, oof_pred, labels=np.arange(n_classes))
    print(f"CV mean logloss: {np.mean(fold_scores):.6f}")
    print(f"OOF logloss: {overall_logloss:.6f}")

    return {
        "label_encoder": label_encoder,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "fold_scores": fold_scores,
        "overall_logloss": overall_logloss
    }


# =========================================================
# 8. Save Outputs
# =========================================================
def save_outputs(train_df, test_df, result):
    print("Saving outputs...")
    class_names = list(result["label_encoder"].classes_)

    oof_df = pd.DataFrame(result["oof_pred"], columns=class_names)
    oof_df.insert(0, ID_COL, train_df[ID_COL].values)
    oof_df.insert(1, GROUP_COL, train_df[GROUP_COL].values)
    oof_df["true_mode"] = train_df[TARGET_COL].values
    oof_df.to_csv("oof_pred_lgb.csv", index=False)

    if test_df is not None and result["test_pred"] is not None:
        test_pred_df = pd.DataFrame(result["test_pred"], columns=class_names)
        test_pred_df.insert(0, ID_COL, test_df[ID_COL].values)
        test_pred_df.to_csv("test_pred_lgb.csv", index=False)

        submission = test_pred_df.copy()
        submission[class_names] = submission[class_names].div(submission[class_names].sum(axis=1), axis=0)
        submission.to_csv("submission_lgb.csv", index=False)

    summary_df = pd.DataFrame({
        "fold": list(range(1, len(result["fold_scores"]) + 1)),
        "logloss": result["fold_scores"]
    })
    summary_df.loc[len(summary_df)] = ["mean", np.mean(result["fold_scores"])]
    summary_df.loc[len(summary_df)] = ["std", np.std(result["fold_scores"])]
    summary_df.loc[len(summary_df)] = ["oof", result["overall_logloss"]]
    summary_df.to_csv("cv_summary_lgb.csv", index=False)
    print("Files saved successfully.")


# =========================================================
# 9. Main Execution
# =========================================================
def main():
    train, test, has_train, has_test = load_and_merge()
    if not has_train:
        print("Pipeline requires training data. Exiting.")
        return

    train, test = handle_missing_values(train, test, has_train, has_test)

    train = logical_quality_check(train)
    train = static_feature_engineering(train)

    if has_test:
        test = logical_quality_check(test)
        test = static_feature_engineering(test)

    result = run_groupkfold_lgb(train, test)
    save_outputs(train, test, result)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()