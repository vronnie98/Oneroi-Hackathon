import pandas as pd
import numpy as np
import re
import os
import warnings
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Data Loading and Merging
# ==========================================
print("Loading and merging data...")

train_files = ["train_val_households.csv", "train_val_persons.csv", "train_val_trips.csv"]
test_files = ["test_households.csv", "test_persons.csv", "test_trips.csv"]

has_train = all(os.path.exists(f) for f in train_files)
has_test = all(os.path.exists(f) for f in test_files)

if not has_train and not has_test:
    raise FileNotFoundError("Error: Missing both train and test files.")

train, test = None, None

if has_train:
    print("Merging train data...")
    train_households = pd.read_csv("train_val_households.csv")
    train_persons = pd.read_csv("train_val_persons.csv")
    train_trips = pd.read_csv("train_val_trips.csv")
    train = train_trips.merge(train_persons, on=["persid", "hhid"], how="left") \
        .merge(train_households, on="hhid", how="left")
else:
    print("Warning: Train data missing. Skipping.")

if has_test:
    print("Merging test data...")
    test_households = pd.read_csv("test_households.csv")
    test_persons = pd.read_csv("test_persons.csv")
    test_trips = pd.read_csv("test_trips.csv")
    test = test_trips.merge(test_persons, on=["persid", "hhid"], how="left") \
        .merge(test_households, on="hhid", how="left")
else:
    print("Warning: Test data missing. Skipping.")

# ==========================================
# 2. Missing Value Handling
# ==========================================
print("Handling missing values...")

ref_df = train if has_train else test

cat_cols = ref_df.select_dtypes(include=["object", "string"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "mode"]

num_cols = ref_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["tripid", "hhid", "persid"]]

# Fill categorical missing values with 'Missing'
for col in cat_cols:
    if has_train and col in train.columns:
        train[col] = train[col].fillna("Missing")
    if has_test and col in test.columns:
        test[col] = test[col].fillna("Missing")

# Fill numerical missing values with median
for col in num_cols:
    if has_train and col in train.columns:
        median_value = train[col].median()
    else:
        median_value = test[col].median()
        print(f"Warning: Using test median for '{col}'.")

    if has_train and col in train.columns:
        train[col] = train[col].fillna(median_value)
    if has_test and col in test.columns:
        test[col] = test[col].fillna(median_value)

# ==========================================
# 3. Logical Quality Check
# ==========================================
print("Executing logical quality checks...")


def logical_quality_check(df):
    df_checked = df.copy()

    # WFH without employment anomaly
    if 'anywork' in df_checked.columns and 'anywfh' in df_checked.columns:
        df_checked['flag_wfh_anomaly'] = ((df_checked['anywork'] == 'No') &
                                          (df_checked['anywfh'] == 'Yes')).astype(int)

    # Implausible speed anomaly (>120 km/h)
    if 'cumdist' in df_checked.columns and 'travtime' in df_checked.columns:
        safe_travtime = df_checked['travtime'].clip(lower=1)
        df_checked['speed_kmh'] = df_checked['cumdist'] / (safe_travtime / 60.0)
        df_checked['flag_speed_anomaly'] = (df_checked['speed_kmh'] > 120).astype(int)

    return df_checked


if has_train: train = logical_quality_check(train)
if has_test:  test = logical_quality_check(test)

# ==========================================
# 4. Feature Engineering
# ==========================================
print("Applying feature engineering...")


def feature_engineering(df_to_transform, df_train_reference=None):
    df_feat = df_to_transform.copy()
    if df_train_reference is None:
        df_train_reference = df_feat

    # Reduce WFH features
    wfh_cols = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri', 'wfhsat', 'wfhsun']
    if all(col in df_feat.columns for col in wfh_cols):
        df_feat['wfh_days_per_week'] = (df_feat[wfh_cols] == 'Yes').sum(axis=1)
        df_feat.drop(columns=wfh_cols, inplace=True)

    # Convert age brackets to median values
    def extract_age_median(age_str):
        if pd.isna(age_str) or age_str == "Missing": return np.nan
        nums = re.findall(r'\d+', str(age_str))
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2.0
        elif len(nums) == 1:
            return float(nums[0])
        return np.nan

    age_cols = ['youngestgroup_5', 'aveagegroup_5', 'oldestgroup_5']
    if all(col in df_feat.columns for col in age_cols):
        df_feat['youngest_age_num'] = df_feat['youngestgroup_5'].apply(extract_age_median)
        df_feat['avg_age_num'] = df_feat['aveagegroup_5'].apply(extract_age_median)
        df_feat['oldest_age_num'] = df_feat['oldestgroup_5'].apply(extract_age_median)

        # Create binary flags for kids and seniors
        df_feat['has_kids'] = (df_feat['youngest_age_num'] < 15).astype(int)
        df_feat['has_seniors'] = (df_feat['oldest_age_num'] >= 65).astype(int)
        df_feat.drop(columns=age_cols, inplace=True)

    # Data-driven time binning
    if 'startime' in df_feat.columns and 'dayType' in df_feat.columns:
        pub_am_mask = (df_train_reference.get('mode', pd.Series()) == 'PUBLICTRANSPORT') & (
                    df_train_reference['startime'] < 720)
        pub_am_times = df_train_reference.loc[pub_am_mask, 'startime'] if not pub_am_mask.empty else pd.Series(
            dtype=float)
        am_peak_start = pub_am_times.quantile(0.15) if not pub_am_times.empty else 390
        am_peak_end = pub_am_times.quantile(0.85) if not pub_am_times.empty else 660

        pub_pm_mask = (df_train_reference.get('mode', pd.Series()) == 'PUBLICTRANSPORT') & (
                    df_train_reference['startime'] >= 720)
        pub_pm_times = df_train_reference.loc[pub_pm_mask, 'startime'] if not pub_pm_mask.empty else pd.Series(
            dtype=float)
        pm_peak_start = pub_pm_times.quantile(0.15) if not pub_pm_times.empty else 780
        pm_peak_end = pub_pm_times.quantile(0.85) if not pub_pm_times.empty else 1080

        edu_pm_mask = (df_train_reference.get('trippurp', pd.Series()) == 'Education') & (
                    df_train_reference['startime'] >= 720)
        if 'trippurp' in df_train_reference.columns:
            edu_pm_times = df_train_reference.loc[edu_pm_mask, 'startime']
            school_peak_start = edu_pm_times.quantile(0.10) if not edu_pm_times.empty else 900
            school_peak_end = edu_pm_times.quantile(0.90) if not edu_pm_times.empty else 960
        else:
            school_peak_start, school_peak_end = 900, 960

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
                if am_peak_start <= mins <= am_peak_end:
                    return 'Morning_Commute_Peak'
                elif school_peak_start <= mins <= school_peak_end:
                    return 'School_Pickup_Peak'
                elif pm_peak_start <= mins <= pm_peak_end:
                    return 'Evening_Commute_Peak'
                elif mins < am_peak_start:
                    return 'Night_Early_Morning'
                elif am_peak_end < mins < school_peak_start:
                    return 'Midday_OffPeak'
                else:
                    return 'Late_Evening'

        df_feat['time_of_day'] = df_feat.apply(assign_datadriven_time, axis=1)

    # Mobility cross-features
    if 'carlicence' in df_feat.columns and 'totalvehs' in df_feat.columns:
        has_licence = df_feat['carlicence'].isin(['Full Licence', 'Probationary Licence']).astype(int)
        df_feat['veh_per_driver'] = df_feat['totalvehs'] / (has_licence + 0.001)

    # Drop redundant columns
    cols_to_drop = ['activities', 'starthour', 'arrhour', 'travtime', 'homeregion_ASGS']
    df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns], inplace=True)

    return df_feat


train_feat = None
test_feat = None

if has_train:
    train_feat = feature_engineering(train)
if has_test:
    reference_df = train if has_train else test
    if not has_train:
        print("Warning: Missing train reference, using test data for time bins.")
    test_feat = feature_engineering(test, df_train_reference=reference_df)

# ==========================================
# 5. Data Prep & Encoding
# ==========================================
print("Preparing model matrix and encoding...")


def prepare_for_model(df, is_train=True):
    y = None
    le = LabelEncoder()

    # Encode target variable
    if is_train and 'mode' in df.columns:
        y = le.fit_transform(df['mode'])
        df = df.drop(columns=['mode'])

    # Convert object/string columns to category dtype
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # Drop ID columns
    X = df.drop(columns=['tripid', 'hhid', 'persid'], errors='ignore')
    groups = df['hhid'] if 'hhid' in df.columns else None

    if is_train:
        return X, y, groups, le
    else:
        return X, df[['tripid']] if 'tripid' in df.columns else None


if has_train:
    X_train, y_train, groups, label_encoder = prepare_for_model(train_feat, is_train=True)
    train_feat.to_csv("processed_train_wide.csv", index=False)
    print("Train dataset saved: processed_train_wide.csv")

if has_test:
    X_test, df_tripid = prepare_for_model(test_feat, is_train=False)
    test_feat.to_csv("processed_test_wide.csv", index=False)
    print("Test dataset saved: processed_test_wide.csv")

print("Data processing pipeline completed.")