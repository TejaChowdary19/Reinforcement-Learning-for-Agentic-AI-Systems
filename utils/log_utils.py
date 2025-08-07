# utils/log_utils.py

import csv
import os

def init_logger():
    os.makedirs("logs", exist_ok=True)
    with open("logs/metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "loss"])

def log_metrics(episode, reward, loss):
    with open("logs/metrics.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward, loss])
