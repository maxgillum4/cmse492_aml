"""
preprocessing_pipeline.py
Adapted from Chapter 2 of *Hands-On Machine Learning with Scikit-Learn,
Keras & TensorFlow, 3rd ed.* (Aurélien Géron).

Steps:
1. Load raw training data
2. Separate labels
3. Build preprocessing pipeline:
   - impute missing values (median)
   - scale numeric attributes
   - one-hot encode categorical attribute
4. Save processed dataset to /data/train
"""

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT = Path("..").resolve()
TRAIN_PATH = ROOT / "data" / "train" / "housing_train.csv"
OUTPUT_PATH = ROOT / "data" / "train" / "housing_train_processed.csv"

# 1. load
housing = pd.read_csv(TRAIN_PATH)

# 2. separate labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 3. numeric vs categorical
num_attribs = list(housing.drop("ocean_proximity", axis=1).columns)
cat_attribs = ["ocean_proximity"]

# 4. pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# rebuild into DataFrame (optional)
housing_prepared_df = pd.DataFrame(
    housing_prepared.toarray() if hasattr(housing_prepared, "toarray") else housing_prepared
)
housing_prepared_df["median_house_value"] = housing_labels.values

# 5. save
housing_prepared_df.to_csv(OUTPUT_PATH, index=False)
print("Processed train saved:", OUTPUT_PATH)
