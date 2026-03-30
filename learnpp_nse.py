import numpy as np
from river import tree


class LearnPPNSE:
    def __init__(self, a=0.5, b=5, max_size=20):
        self.models = []
        self.beta_history = []
        self.voting_weights = []
        self.a = a
        self.b = b
        self.max_size = max_size
        self.classes_ = None

    # -------------------------------------------------
    # Entrenamiento por chunk
    # -------------------------------------------------
    def fit_chunk(self, X, y):

        X_dict = X.to_dict(orient="records")

        # ---------------------------------------------
        # 1. Evaluar ensemble actual (E_t)
        # ---------------------------------------------
        if len(self.models) > 0:

            errors = []

            for xi, yi in zip(X_dict, y):
                votes = {}

                for model, w in zip(self.models, self.voting_weights):
                    pred = model.predict_one(xi)
                    votes[pred] = votes.get(pred, 0) + w

                pred_final = max(votes, key=votes.get)
                errors.append(pred_final != yi)

            E_t = np.mean(errors)

        else:
            E_t = 0.5  # inicialización razonable

        # ---------------------------------------------
        # 2. Construir distribución D_t
        # ---------------------------------------------
        weights = []

        for xi, yi in zip(X_dict, y):

            if len(self.models) > 0:
                votes = {}

                for model, w in zip(self.models, self.voting_weights):
                    pred = model.predict_one(xi)
                    votes[pred] = votes.get(pred, 0) + w

                pred_final = max(votes, key=votes.get)

                if pred_final == yi:
                    w = E_t
                else:
                    w = 1.0
            else:
                w = 1.0

            weights.append(w)

        weights = np.array(weights)
        D_t = weights / (weights.sum() + 1e-10)

        # ---------------------------------------------
        # 3. Entrenar nuevo clasificador
        # ---------------------------------------------
        new_model = tree.HoeffdingTreeClassifier()

        for xi, yi in zip(X_dict, y):
            new_model.learn_one(xi, yi)

        self.models.append(new_model)
        self.beta_history.append([])

        # limitar tamaño
        if len(self.models) > self.max_size:
            self.models.pop(0)
            self.beta_history.pop(0)

        if self.classes_ is None:
            self.classes_ = np.unique(y)

        # ---------------------------------------------
        # 4. Evaluar TODOS los modelos con D_t
        # ---------------------------------------------
        for k, model in enumerate(self.models):

            error = 0

            for i, (xi, yi) in enumerate(zip(X_dict, y)):
                if model.predict_one(xi) != yi:
                    error += D_t[i]

            # clipping como en el paper
            error = max(min(error, 0.5), 1e-6)

            beta = error / (1 - error + 1e-10)

            self.beta_history[k].append(beta)

        # ---------------------------------------------
        # 5. Voting weights
        # ---------------------------------------------
        self.voting_weights = []

        for beta_hist in self.beta_history:

            T = len(beta_hist)

            omega = []
            for j in range(T):
                age = T - j - 1
                weight = 1 / (1 + np.exp(self.a * (age - self.b)))
                omega.append(weight)

            omega = np.array(omega)
            omega = omega / (omega.sum() + 1e-10)

            beta_avg = np.sum(omega * np.array(beta_hist))

            weight = np.log(1 / (beta_avg + 1e-10))

            self.voting_weights.append(weight)
    # -------------------------------------------------
    # Predicción
    # -------------------------------------------------
    def predict(self, X):

        if not self.models:
            raise ValueError("El ensemble está vacío.")

        X_dict = X.to_dict(orient="records")
        predictions = []

        for xi in X_dict:

            votes = {}

            for model, w in zip(self.models, self.voting_weights):
                pred = model.predict_one(xi)
                votes[pred] = votes.get(pred, 0) + w

            predictions.append(max(votes, key=votes.get))

        return np.array(predictions)