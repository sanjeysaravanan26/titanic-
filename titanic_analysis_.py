import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot settings
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (7, 5)
})

print("Libraries loaded ✓")

# ── 1 · Load Dataset ──────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("titanic.csv")

print(f"\nShape: {df.shape}")
print(df.head())

# ── 2 · Explore & Clean Data ─────────────────────────────────────────────────

print("\n── Dataset Info ──")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ── 3 · Analysis Questions ───────────────────────────────────────────────────

print("\n── Q1: Survival Rate by Gender ──")
survival_by_gender = df.groupby("Sex")["Survived"].mean() * 100
print(survival_by_gender.round(2))

print("\n── Q2: Survival Rate by Passenger Class ──")
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100
print(survival_by_class.round(2))

# ── 4 · Visualizations ───────────────────────────────────────────────────────

# 4a · Survival Rate by Gender
plt.figure()
ax = sns.barplot(x="Sex", y="Survived", data=df, palette="viridis", errorbar=None)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")

ax.set_title("Survival Rate by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "survival_by_gender.png"), dpi=130)
plt.close()
print("\nSaved: survival_by_gender.png ✓")

# 4b · Survival Rate by Passenger Class
plt.figure()
ax = sns.barplot(x="Pclass", y="Survived", data=df, palette="magma", errorbar=None)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")

ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "survival_by_class.png"), dpi=130)
plt.close()
print("Saved: survival_by_class.png ✓")

# 4c · Distribution of Passenger Ages
plt.figure()
plt.hist(df["Age"], bins=20, color="#4C72B0", edgecolor="black")

plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_histogram.png"), dpi=130)
plt.close()
print("Saved: age_histogram.png ✓")

print("\n✅ All analysis complete!")
