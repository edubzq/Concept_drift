import numpy as np
from river import tree
import pandas as pd


class SEAEnsemble:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.models = []

    # ----------------------------------------
    # Entrenamiento por chunk
    # ----------------------------------------
    def fit_chunk(self, X, y):

        # Crear nuevo Hoeffding Tree
        model = tree.HoeffdingTreeClassifier()

        # Entrenamiento incremental instancia a instancia
        for xi, yi in zip(X.to_dict(orient="records"), y):
            model.learn_one(xi, yi)

        # Añadir modelo
        self.models.append(model)

        # Si excede tamaño → eliminar peor
        if len(self.models) > self.max_size:
            subset = X.sample(frac=0.5)
            subset = X.sample(frac=0.5)
            y_series = pd.Series(y, index=X.index)
            y_subset = y_series.loc[subset.index].values

            self._remove_worst(subset, y_subset)
    # ----------------------------------------
    # Eliminar el modelo con peor accuracy
    # ----------------------------------------
    def _remove_worst(self, X, y):

        accuracies = []

        for model in self.models:
            correct = 0
            for xi, yi in zip(X.to_dict(orient="records"), y):
                pred = model.predict_one(xi)
                if pred == yi:
                    correct += 1

            acc = correct / len(y)
            accuracies.append(acc)

        # Eliminar modelo con peor accuracy
        worst_index = np.argmin(accuracies)
        del self.models[worst_index]

    # ----------------------------------------
    # Voting simple (majority vote)
    # ----------------------------------------
    def predict(self, X):

        predictions = []

        for xi in X.to_dict(orient="records"):

            votes = {}

            for model in self.models:
                pred = model.predict_one(xi)
                votes[pred] = votes.get(pred, 0) + 1

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)