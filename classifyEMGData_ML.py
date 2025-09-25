import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# Load Labeled Dataset
df = pd.read_csv("ekg_data.csv")

if "record_id" not in df.columns:
    raise ValueError("CSV must contain a 'record_id' column to group signals")

# Feature Extraction
features = []
labels = []

for rec_id, group in df.groupby("record_id"):
    signal = group["amplitude"].values
    features.append([np.mean(signal), np.std(signal),
                     np.max(signal), np.min(signal), np.ptp(signal)])
    labels.append(group["label"].iloc[0])

X = np.array(features)
y = np.array(labels)

# Cross validation setup
class_counts = Counter(y)
min_class_samples = min(class_counts.values())
n_splits = min(5, min_class_samples)

if n_splits < 2:
    raise ValueError("Not enough samples per class for cross-validation")

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

fold = 1
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\nFold {fold} Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    fold += 1

print("\nOverall Cross-Validation Metrics:")
print(classification_report(all_y_true, all_y_pred, zero_division=0))
print("Overall Accuracy:", accuracy_score(all_y_true, all_y_pred))

# Final trained model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Load and extract features from new unlabeled data
df_new = pd.read_csv("new_ekg_data.csv")
features_new = []
record_ids = []

for rec_id, group in df_new.groupby("record_id"):
    signal = group["amplitude"].values
    features_new.append([np.mean(signal), np.std(signal),
                         np.max(signal), np.min(signal), np.ptp(signal)])
    record_ids.append(rec_id)

X_new = np.array(features_new)

# Label Prediction
predicted_labels = final_model.predict(X_new)

results = pd.DataFrame({
    "record_id": record_ids,
    "predicted_label": predicted_labels
})

print("\nPredictions for new data:")
print(results)

# Optionally save to CSV
results.to_csv("predicted_labels.csv", index=False)
