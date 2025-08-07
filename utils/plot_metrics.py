# utils/plot_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
file_path = "logs/metrics.csv"
if not os.path.exists(file_path):
    print("❌ metrics.csv not found.")
    exit()

df = pd.read_csv(file_path)
print("✅ Columns in CSV:", df.columns.tolist())

# Rename columns if needed (make it safe to access)
df.columns = [col.strip().capitalize() for col in df.columns]

# Plot reward and loss
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df["Episode"], df["Reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode vs Reward")

plt.subplot(1, 2, 2)
plt.plot(df["Episode"], df["Loss"])
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Episode vs Loss")

plt.tight_layout()
plt.savefig("logs/training_metrics_plot.png")
plt.show()
