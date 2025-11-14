import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Теперь можно импортировать

from model import train_model
from utils import plot


def main():
    metrics_dict, performance_history = train_model(plot_every_n_steps)

    for model_name, metric in metrics_dict.items():
        print(f"  - {model_name}: {metric.get():.4f}")

    plot(performance_history, plot_every_n_steps, drift_point)


if __name__ == "__main__":
    drift_point: int = 2500
    plot_every_n_steps: int = 100

    main()
