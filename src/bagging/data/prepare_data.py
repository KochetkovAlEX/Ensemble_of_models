from river import datasets

DATASET = datasets.synth.SEA(seed=42, variant=1).take(5000)
