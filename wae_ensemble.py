import numpy as np
from river import tree


class WAEEnsemble:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.models = []
        self.iterations = []
        self.weights = []
        self.classes_ = None

    # -------------------------------------------------
    # Entrenamiento por chunk
    # -------------------------------------------------
    def fit_chunk(self, X, y):

        model = tree.HoeffdingTreeClassifier()

        # Entrenamiento incremental
        for xi, yi in zip(X.to_dict(orient="records"), y):
            model.learn_one(xi, yi)

        if self.classes_ is None:
            self.classes_ = np.unique(y)

        self.models.append(model)
        self.iterations.append(1)

        # Pruning si excede tamaño
        if len(self.models) > self.max_size:
            self._diversity_pruning(X, y)

        # Actualizar pesos
        self._update_weights(X, y)

    # -------------------------------------------------
    # Generalized Diversity
    # -------------------------------------------------
    def _compute_gd(self, models_subset, X, y):

        L = len(models_subset)
        if L < 2:
            return 0

        preds = []

        for model in models_subset:
            model_preds = []
            for xi in X.to_dict(orient="records"):
                model_preds.append(model.predict_one(xi))
            preds.append(model_preds)

        preds = np.array(preds)
        N = len(y)

        disagreements = 0
        total_pairs = 0

        for i in range(L):
            for j in range(i + 1, L):
                disagreements += np.sum(preds[i] != preds[j]) / N
                total_pairs += 1

        return disagreements / total_pairs

    # -------------------------------------------------
    # Pruning por diversidad
    # -------------------------------------------------
    def _diversity_pruning(self, X, y):

        best_gd = -1
        remove_index = None

        for i in range(len(self.models)):
            subset = self.models[:i] + self.models[i+1:]
            gd = self._compute_gd(subset, X, y)

            if gd > best_gd:
                best_gd = gd
                remove_index = i

        del self.models[remove_index]
        del self.iterations[remove_index]

    # -------------------------------------------------
    # Actualizar pesos
    # w = accuracy / sqrt(iteraciones)
    # -------------------------------------------------
    def _update_weights(self, X, y):

        self.weights = []

        for i, model in enumerate(self.models):

            correct = 0
            for xi, yi in zip(X.to_dict(orient="records"), y):
                pred = model.predict_one(xi)
                if pred == yi:
                    correct += 1

            acc = correct / len(y)
            weight = acc / np.sqrt(self.iterations[i])

            self.weights.append(weight)
            self.iterations[i] += 1

        # Normalizar
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    # -------------------------------------------------
    # Voting ponderado multiclase
    # -------------------------------------------------
    def predict(self, X):

        if not self.models:
            raise ValueError("El ensemble está vacío.")

        predictions = []

        for xi in X.to_dict(orient="records"):

            votes = {}

            for model, w in zip(self.models, self.weights):
                pred = model.predict_one(xi)
                votes[pred] = votes.get(pred, 0) + w

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)