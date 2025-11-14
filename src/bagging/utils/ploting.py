from matplotlib import pyplot as plt


def plot(performance_history: dict, plot_every_n_steps=100, drift_point=2500):
    fig, ax = plt.subplots(figsize=(12, 7))

    x_axis = range(plot_every_n_steps, 5000 + 1, plot_every_n_steps)

    for model_name, history in performance_history.items():
        ax.plot(x_axis, history, label=model_name, marker="o", markersize=3, alpha=0.8)

    ax.axvline(
        drift_point,
        color="red",
        linestyle="--",
        label=f"Дрейф концепта (на {drift_point})",
    )

    ax.set_title(
        "Сравнение производительности онлайн-ансамблей в условиях дрейфа концепта"
    )
    ax.set_xlabel("Количество обработанных примеров")
    ax.set_ylabel("Точность (Prequential Accuracy)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 1.05)
    plt.show()
