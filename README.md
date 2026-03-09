1. Data Integration & Missing Value Handling
Hierarchical Merging: Survey data naturally follows a "Household -> Person -> Trip" hierarchy. The system uses the Trips table as the base and performs left joins using hhid and persid to construct a comprehensive wide table.

Anti-Leakage Imputation Strategy:

Categorical Features: Filled uniformly with "Missing", treating the absence of data as an independent, valid category.

Numerical Features: Strictly imputed using the median of the training set. When processing the test set, the training medians are enforced to prevent data leakage from the test distribution.

2. Logical Quality Check & Anomaly Flagging
Survey data often contains reporting errors. Instead of deleting dirty data (which risks model failure when encountering similar anomalies in the test set), this pipeline converts them into anomaly flags for the model to learn.

Business Logic Conflict (flag_wfh_anomaly): If a respondent claims to have "no work" but also reports "working from home," this contradiction is captured as a binary feature.

Physical Law Conflict (flag_speed_anomaly): Calculates physical travel speed (speed_kmh) using "cumulative distance / travel time". Speeds exceeding 120km/h are flagged as anomalies. This allows the model to implicitly identify implausible WALK/CYCLE samples without needing the target variable.

3. Feature Engineering & Dimensionality Reduction
This module eliminates multicollinearity and compresses high-dimensional, sparse features into high-information-density predictors.

WFH Dimensionality Reduction: Aggregates the 7 independent binary variables for Monday-Sunday into a single numerical feature wfh_days_per_week, directly reflecting the intensity of remote work.

Age Bracket Digitization: Converts non-comparable string ranges (e.g., "15->19") into continuous medians (e.g., 17.5) using regular expressions. It simultaneously derives strong domain labels: has_kids (under 15) and has_seniors (over 65).

Data-Driven Time Binning:

Philosophy: Discards arbitrary time boundary guesses. It relies entirely on the quantiles of specific trip purposes within the training set to define true peak hours.

Implementation: Extracts the 15th-85th percentiles of "Public Transport" trips to define morning and evening commuter peaks, and the 10th-90th percentiles of "Education" trips for school pickup peaks. Weekends use broader, static bins.

Mobility Cross-Features: Constructs veh_per_driver (total household vehicles / household members with valid licenses) to quantify the actual competition for vehicle availability during a trip.

Redundancy Elimination: Drops low-variance columns (e.g., activities), perfectly collinear features (e.g., starthour, travtime), and overly broad spatial groupings (homeregion_ASGS).

4. Model Preparation & Encoding
Native Category Support: Avoids One-Hot Encoding, which causes feature space explosion. All text columns are converted to Pandas category dtype to trigger LightGBM's native histogram-based split optimization.

ID Stripping: Removes tripid, hhid, and persid before outputting the final feature matrix X to prevent the model from cheating by memorizing identifiers, thereby improving generalization.