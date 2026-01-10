import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load training data (already standardized by your generator)
df = pd.read_csv("data/raw/rain_supervised_train.csv")

X = df.drop(columns=["label"])
y = df["label"].map({"NO": 0, "YES": 1}).astype(int)

# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predicted probability for YES (rain tomorrow)
p_yes = model.predict_proba(X)[:, 1]

# For visualization, we want humidity in the SAME units as the CSV:
# Note: Your CSV is standardized (z-scores), so humidity_pct here is "standard deviations from mean".
hum_z = X["humidity_pct"].to_numpy()

# Smooth curve: bin humidity and average predicted probabilities
n_bins = 40
bins = np.linspace(hum_z.min(), hum_z.max(), n_bins + 1)
bin_idx = np.digitize(hum_z, bins) - 1

bin_centers = []
bin_mean_p = []
bin_counts = []

for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() < 30:  # skip tiny bins to reduce noise
        continue
    bin_centers.append(hum_z[mask].mean())
    bin_mean_p.append(p_yes[mask].mean())
    bin_counts.append(mask.sum())

bin_centers = np.array(bin_centers)
bin_mean_p = np.array(bin_mean_p)

# Plot
plt.figure()
plt.scatter(hum_z, p_yes, alpha=0.12, s=10)
plt.plot(bin_centers, bin_mean_p, linewidth=2)

plt.xlabel("humidity_pct (standardized z-score)")
plt.ylabel("Predicted P(rain tomorrow = YES)")
plt.title("Humidity vs predicted probability of rain tomorrow")

plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
