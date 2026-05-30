import numpy as np
from river import tree


class WAEEnsemble:
    def __init__(self, max_size=20, accuracy_weight=0.7, diversity_weight=0.3):
        self.max_size = max_size
        self.accuracy_weight = accuracy_weight
        self.diversity_weight = diversity_weight
        self.models = []
        self.iterations = []
        self.weights = []
        self.classes_ = None

    # -------------------------------------------------
    # Entrenamiento por chunk
    # -------------------------------------------------
    def fit_chunk(self, X, y):
        X_records = X.to_dict(orient="records")
        y_array = np.asarray(y)

        model = tree.HoeffdingTreeClassifier()

        # Entrenamiento incremental
        for xi, yi in zip(X_records, y_array):
            model.learn_one(xi, yi)

        if self.classes_ is None:
            self.classes_ = np.unique(y_array)

        self.models.append(model)
        self.iterations.append(1)

        base_predictions = self._predict_base_matrix(X_records)

        # Pruning si excede tamaño
        if len(self.models) > self.max_size:
            remove_index = self._choose_pruning_index(base_predictions, y_array)
            del self.models[remove_index]
            del self.iterations[remove_index]
            base_predictions = np.delete(base_predictions, remove_index, axis=0)

        # Actualizar pesos
        self._update_weights_from_predictions(base_predictions, y_array)

    # -------------------------------------------------
    # Matriz de predicciones base
    # -------------------------------------------------
    def _predict_base_matrix(self, X_records):
        return np.array(
            [[model.predict_one(xi) for xi in X_records] for model in self.models],
            dtype=object,
        )

    # -------------------------------------------------
    # Generalized Diversity / disagreement medio
    # -------------------------------------------------
    def _compute_gd_from_predictions(self, preds):
        n_models = preds.shape[0]
        if n_models < 2:
            return 0.0

        disagreements = 0.0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += float(np.mean(preds[i] != preds[j]))
                total_pairs += 1

        return disagreements / total_pairs

    # -------------------------------------------------
    # Pruning por balance accuracy-diversidad
    # -------------------------------------------------
    def _choose_pruning_index(self, base_predictions, y):
        best_score = -np.inf
        remove_index = 0

        for i in range(len(self.models)):
            subset_preds = np.delete(base_predictions, i, axis=0)
            remaining_accuracies = np.mean(subset_preds == y, axis=1)
            mean_accuracy = float(np.mean(remaining_accuracies))
            diversity = self._compute_gd_from_predictions(subset_preds)

            score = (
                self.accuracy_weight * mean_accuracy
                + self.diversity_weight * diversity
            )

            if score > best_score:
                best_score = score
                remove_index = i

        return remove_index

    # -------------------------------------------------
    # Actualizar pesos
    # w = accuracy / sqrt(iteraciones)
    # -------------------------------------------------
    def _update_weights_from_predictions(self, base_predictions, y):
        accuracies = np.mean(base_predictions == y, axis=1)
        self.weights = []

        for i, acc in enumerate(accuracies):
            weight = float(acc) / np.sqrt(self.iterations[i])
            self.weights.append(weight)
            self.iterations[i] += 1

        # Normalizar
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
        elif self.weights:
            self.weights = [1.0 / len(self.weights)] * len(self.weights)

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