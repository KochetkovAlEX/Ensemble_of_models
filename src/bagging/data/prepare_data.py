from river import datasets


def load_dataset():
    return datasets.synth.SEA(seed=42, variant=1).take(5000)
