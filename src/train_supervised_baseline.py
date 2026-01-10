import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/raw/rain_supervised_train.csv")

# Separate features and label
X = df.drop(columns=["label"])
y = df["label"].map({"NO": 0, "YES": 1})

# Train/validation split (inside training set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a very simple model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["NO", "YES"]))

# Show which features matter most
importance = pd.Series(
    model.coef_[0],
    index=X.columns
).sort_values(key=abs, ascending=False)

print("\nTop features influencing rain prediction:")
print(importance.head(10))
