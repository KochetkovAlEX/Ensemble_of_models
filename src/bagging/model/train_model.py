from data import load_dataset
from models import load_models
from river import metrics

DATASET = load_dataset()
MODELS = load_models()


def train_model(plot_every_n_steps: int = 100):
    metrics_dict = {name: metrics.Accuracy() for name in MODELS.keys()}

    performance_history = {name: [] for name in MODELS.keys()}

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

    return metrics_dict, performance_history
