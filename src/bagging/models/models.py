from river import ensemble, forest, tree


def load_models():
    return {
        "Одиночное Дерево (HT)": tree.HoeffdingTreeClassifier(),
        "Обычный Бэггинг (Bagging)": ensemble.LeveragingBaggingClassifier(
            n_models=10, model=tree.HoeffdingTreeClassifier(), seed=42
        ),
        "Адаптивный Бэггинг (Leveraging Bagging)": ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42
        ),
        # "Случайный Лес (Random Forest)": forest.adaptive_random_forest.BaseTreeClassifier(
        #     max_depth=10
        # ),
        # "Случайный Адаптивный Лес (Adaptive Random Forest)": forest.adaptive_random_forest.ARFClassifier(
        #     n_models=10, seed=42, max_depth=10
        # ),
    }
