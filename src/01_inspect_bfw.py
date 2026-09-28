import pandas as pd
from config import BFW_DATA_PATH

# Load dataset
df = pd.read_csv(BFW_DATA_PATH)

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== LABEL COUNTS ===")
print(df["label"].value_counts())

print("\n=== FOLDS ===")
print(df["fold"].value_counts().sort_index())

print("\n=== RACE × GENDER COUNTS ===")
print(df["a1"].value_counts())

print("\n=== GENDER COUNTS ===")
print(df["g1"].value_counts())

print("\n=== RACE × GENDER ===")
print(pd.crosstab(df["a1"], df["g1"]))

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== MODEL SCORE SUMMARY ===")
print(df[["vgg16", "resnet50", "senet50"]].describe())