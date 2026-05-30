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

    # Entrenar nuevo modelo
        new_model = tree.HoeffdingTreeClassifier()

        for xi, yi in zip(X.to_dict(orient="records"), y):
            new_model.learn_one(xi, yi)

        # Añadir nuevo modelo
        self.models.append(new_model)

        # Recalcular pesos de TODOS los modelos
        self.weights = []

        for model in self.models:
            error = self._compute_error(model, X, y)
            weight = 1 / (error + self.epsilon)
            self.weights.append(weight)

        # Si excede tamaño máximo, mantener los mejores
        if len(self.models) > self.max_size:
            self._keep_top_k()

    # ----------------------------------------
    # Calcular error rate
    # ----------------------------------------
    def _compute_error(self, model, X, y):

        wrong = 0

        for xi, yi in zip(X.to_dict(orient="records"), y):
            pred = model.predict_one(xi)
            if pred != yi:
                wrong += 1

        return wrong / len(y)

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

        predictions = []

        for xi in X.to_dict(orient="records"):

            votes = {}

            for model, w in zip(self.models, self.weights):
                pred = model.predict_one(xi)
                votes[pred] = votes.get(pred, 0) + w

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)