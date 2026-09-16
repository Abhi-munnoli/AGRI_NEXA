import os
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "AgriNexa_Hukkeri_Yield_Dataset_2015_2025.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)


# =========================================================
# 2. INPUT FEATURES
# =========================================================
# IMPORTANT:
# These are exactly the 8 inputs used by your application.

FEATURES = [
    "Crop",
    "Soil_Type",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_percent",
    "Soil_Moisture_percent",
    "Soil_pH",
    "Area_acre"
]

TARGET = "Yield_ton_ha"


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\n" + "=" * 60)
print("        YIELDNEXT MODEL TRAINING")
print("=" * 60)

if not os.path.exists(DATASET):
    print("\nERROR: Dataset not found!")
    print("\nExpected file:")
    print(DATASET)
    print("\nPut the CSV file in the same folder as train_model.py.")
    exit()

print("\nLoading dataset...")

data = pd.read_csv(DATASET)

print("Dataset loaded successfully.")
print("Rows:", len(data))
print("Columns:", list(data.columns))


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    print(missing_columns)

    print("\nYour dataset must contain:")
    for column in required_columns:
        print("-", column)

    exit()


# =========================================================
# 5. REMOVE EMPTY DATA
# =========================================================

data = data.dropna(
    subset=required_columns
).copy()

print("\nRows after cleaning:", len(data))


# =========================================================
# 6. INPUT AND OUTPUT
# =========================================================

X = data[FEATURES]

y = data[TARGET]


# =========================================================
# 7. CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Crop",
    "Soil_Type"
]


# =========================================================
# 8. NUMERICAL FEATURES
# =========================================================

numeric_features = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_percent",
    "Soil_Moisture_percent",
    "Soil_pH",
    "Area_acre"
]


# =========================================================
# 9. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )
    ]
)


# =========================================================
# 10. RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(

    n_estimators=350,

    max_depth=18,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


# =========================================================
# 11. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# =========================================================
# 12. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 13. TRAIN MODEL
# =========================================================

print("\nStarting training...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# =========================================================
# 14. TEST MODEL
# =========================================================

print("\nTesting model...")

predictions = pipeline.predict(
    X_test
)


# =========================================================
# 15. MODEL PERFORMANCE
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# 16. SAVE MODEL
# =========================================================

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )


# =========================================================
# 17. RESULTS
# =========================================================

print("\n" + "=" * 60)
print("       MODEL TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nMAE       : {mae:.3f} ton/ha"
)

print(
    f"RMSE      : {rmse:.3f} ton/ha"
)

print(
    f"R2 Score  : {r2:.3f}"
)

print(
    "\nModel saved successfully:"
)

print(
    MODEL_PATH
)

print("\n" + "=" * 60)
print("IMPORTANT")
print("=" * 60)

print("""
The trained model uses exactly these 8 inputs:

1. Crop
2. Soil Type
3. Rainfall
4. Temperature
5. Humidity
6. Soil Moisture
7. Soil pH
8. Area (acre)

It does NOT use:

- Nitrogen
- Phosphorus
- Potassium
- Fertilizer quantity
""")

print("=" * 60)