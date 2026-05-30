import numpy as np
from river import tree


class AUE2Ensemble:
    def __init__(self, max_size=20, epsilon=1e-8):
        self.max_size = max_size
        self.epsilon = epsilon
        self.models = []
        self.weights = []

    # ----------------------------------------
    # Entrenamiento por chunk
    # ----------------------------------------
    def fit_chunk(self, X, y):
        X_records = X.to_dict(orient="records")
        y_array = np.asarray(y)

        # Recalcular pesos de los modelos existentes antes de entrenar el nuevo
        # clasificador sobre el chunk actual. Así evitamos dar al nuevo modelo un
        # peso optimista por evaluarlo sobre datos que acaba de memorizar.
        if self.models:
            self.weights = [
                self._weight_from_predictions(model, X_records, y_array)
                for model in self.models
            ]

        # Entrenar nuevo modelo
        new_model = tree.HoeffdingTreeClassifier()
        for xi, yi in zip(X_records, y_array):
            new_model.learn_one(xi, yi)

        # El nuevo clasificador entra con peso neutro hasta la siguiente ventana.
        # Esto mantiene la evaluación test-then-train del script principal.
        neutral_weight = float(np.mean(self.weights)) if self.weights else 1.0
        self.models.append(new_model)
        self.weights.append(neutral_weight)

        # Si excede tamaño máximo, mantener los mejores
        if len(self.models) > self.max_size:
            self._keep_top_k()

    # ----------------------------------------
    # Calcular peso por error rate
    # ----------------------------------------
    def _weight_from_predictions(self, model, X_records, y):
        preds = np.array([model.predict_one(xi) for xi in X_records])
        error = float(np.mean(preds != y))
        return 1.0 / (error + self.epsilon)

    # ----------------------------------------
    # Mantener los K mejores modelos
    # ----------------------------------------
    def _keep_top_k(self):
        indices = np.argsort(self.weights)[::-1]  # ordenar descendente
        indices = indices[:self.max_size]

        self.models = [self.models[i] for i in indices]
        self.weights = [self.weights[i] for i in indices]

    # ----------------------------------------
    # Voting ponderado
    # ----------------------------------------
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