import DATASETS
from river import metrics

from bagging.data import DATASET
from models.models import MODELS


def main():
    metrics_dict = {name: metrics.Accuracy() for name in MODELS.keys()}

    performance_history = {name: [] for name in MODELS.keys()}
    plot_every_n_steps = 100

    for i, (x, y) in enumerate(DATASET):
        for model_name, model in MODELS.items():
            y_pred = model.predict_one(x)

            if y_pred is not None:
                metrics_dict[model_name].update(y, y_pred)

            model.learn_one(x, y)

        if (i + 1) % plot_every_n_steps == 0:
            for model_name in MODELS.keys():
                accuracy = metrics_dict[model_name].get()
                performance_history[model_name].append(accuracy)

        print("Итоговая точность:")
        for model_name, metric in metrics_dict.items():
            print(f"  - {model_name}: {metric.get():.4f}")
