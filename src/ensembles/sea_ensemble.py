import numpy as np
import pandas as pd
from river import tree


class SEAEnsemble:
    def __init__(self, max_size=20, pruning_sample_frac=0.5, random_state=42):
        self.max_size = max_size
        self.pruning_sample_frac = pruning_sample_frac
        self.random_state = random_state
        self.models = []

    # ----------------------------------------
    # Entrenamiento por chunk
    # ----------------------------------------
    def fit_chunk(self, X, y):
        X_records = X.to_dict(orient="records")

        # Crear nuevo Hoeffding Tree
        model = tree.HoeffdingTreeClassifier()

        # Entrenamiento incremental instancia a instancia
        for xi, yi in zip(X_records, y):
            model.learn_one(xi, yi)

        # Añadir modelo
        self.models.append(model)

        # Si excede tamaño → eliminar peor
        if len(self.models) > self.max_size:
            subset = X.sample(
                frac=self.pruning_sample_frac,
                random_state=self.random_state,
            )
            y_series = pd.Series(y, index=X.index)
            y_subset = y_series.loc[subset.index].values
            subset_records = subset.to_dict(orient="records")

            self._remove_worst(subset_records, y_subset)

    # ----------------------------------------
    # Eliminar el modelo con peor accuracy
    # ----------------------------------------
    def _remove_worst(self, X_records, y):
        accuracies = []

        for model in self.models:
            preds = np.array([model.predict_one(xi) for xi in X_records])
            accuracies.append(float(np.mean(preds == y)))

        # Eliminar modelo con peor accuracy
        worst_index = int(np.argmin(accuracies))
        del self.models[worst_index]

    # ----------------------------------------
    # Voting simple (majority vote)
    # ----------------------------------------
    def predict(self, X):
        if not self.models:
            raise ValueError("El ensemble está vacío.")

        predictions = []

        for xi in X.to_dict(orient="records"):
            votes = {}

            for model in self.models:
                pred = model.predict_one(xi)
                votes[pred] = votes.get(pred, 0) + 1

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)