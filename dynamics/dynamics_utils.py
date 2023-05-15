import numpy as np

class Stats:
    def __init__(self):
        self.stats = dict()

    def add(self, key, value):
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(value)

    def keys(self):
        return list(self.stats.keys())

    def __getitem__(self, key):
        return self.stats[key]

    def items(self):
        return self.stats.items()