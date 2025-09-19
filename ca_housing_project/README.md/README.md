# CA Housing Project

Course project following Ch. 2 of *Hands-On Machine Learning (3rd ed., Aurélien Géron)*.

## Structure
- `data/raw/`: original `housing.csv` (never modified)
- `data/train/`: `housing_train.csv` (13 cols), `housing_train_processed.csv` (~24 features)
- `data/test/`: `housing_test.csv` (13 cols)
- `images/`: saved plots from IDA/EDA
- `analysis/`: `ida.ipynb`, `eda.ipynb`, `preprocessing_pipeline.py`
- `models/`: `LinearRegression.ipynb`, `DecisionTree.ipynb`, `RandomForest.ipynb`, `SVR.ipynb`

## Reproduce
1. Open `analysis/ida.ipynb` → run to create train/test CSVs and images.
2. Open `analysis/eda.ipynb` → run to create processed training CSV and images.
3. (Optional) `python analysis/preprocessing_pipeline.py` to regenerate processed data via sklearn pipeline.
4. Run each notebook in `models/` to train, cross-validate, tune, and save models to `/models`.

All file paths are relative to repo root.
