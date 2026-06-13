import numpy as np
from river import tree


class LearnPPNSE:

    def __init__(
        self,
        a=0.5,
        b=5,
        max_size=20,
        grace_period=200,
        delta=1e-7,
        pruning_strategy=0,
        weight_power=1.0,
    ):
        self.models = []
        self.beta_history = []
        self.voting_weights = []
        self.a = a
        self.b = b
        self.max_size = max_size
        self.grace_period = int(grace_period)
        self.delta = float(delta)
        self.pruning_strategy = int(pruning_strategy)
        self.weight_power = float(weight_power)
        self.classes_ = None

    def set_config(
        self,
        a=None,
        b=None,
        max_size=None,
        grace_period=None,
        delta=None,
        pruning_strategy=None,
        weight_power=None,
    ):
        """Actualiza la configuración del ensemble sin reiniciar su historial."""
        if a is not None:
            self.a = float(a)
        if b is not None:
            self.b = float(b)
        if max_size is not None:
            self.max_size = int(max_size)
        if grace_period is not None:
            self.grace_period = int(grace_period)
        if delta is not None:
            self.delta = float(delta)
        if pruning_strategy is not None:
            self.pruning_strategy = int(pruning_strategy)
        if weight_power is not None:
            self.weight_power = float(weight_power)

        if len(self.models) > self.max_size:
            self.prune_to_max_size()
        elif self.models:
            self._refresh_voting_weights()

    # -------------------------------------------------
    # Utilidades internas de predicción
    # -------------------------------------------------
    def _predict_base_matrix_from_records(self, X_records):
        """
        Devuelve una matriz (n_modelos, n_instancias) con la predicción de cada
        clasificador base.  evita recalcular las mismas
        predicciones para accuracy, diversidad y actualización de pesos.
        """
        return np.array(
            [[model.predict_one(xi) for xi in X_records] for model in self.models],
            dtype=object,
        )

    def _weighted_vote_from_matrix(self, base_predictions):
        """
        Aplica la votación ponderada sobre una matriz de predicciones base.
        """
        if base_predictions.size == 0:
            return np.array([], dtype=object)

        predictions = []
        weights = self.voting_weights

        for sample_predictions in base_predictions.T:
            votes = {}
            for pred, weight in zip(sample_predictions, weights):
                votes[pred] = votes.get(pred, 0.0) + weight
            predictions.append(max(votes, key=votes.get))

        return np.array(predictions, dtype=object)

    def _refresh_voting_weights(self):
        """
        Recalcula los pesos de votación a partir del historial de betas.
        """
        self.voting_weights = []

        for beta_hist in self.beta_history:
            beta_values = np.asarray(beta_hist, dtype=float)
            n_betas = len(beta_values)

            if n_betas == 0:
                self.voting_weights.append(1.0)
                continue

            ages = np.arange(n_betas - 1, -1, -1, dtype=float)
            omega = 1.0 / (1.0 + np.exp(self.a * (ages - self.b)))
            omega = omega / (omega.sum() + 1e-10)

            beta_avg = float(np.dot(omega, beta_values))
            base_weight = max(np.log(1.0 / (beta_avg + 1e-10)), 0.0)
            weight = base_weight ** self.weight_power

            self.voting_weights.append(weight)

    def _choose_pruning_index(self, base_predictions=None, y=None):
        """Selecciona qué clasificador eliminar según self.pruning_strategy."""
        if len(self.models) == 0:
            raise ValueError("No se puede podar un ensemble vacío.")

        if self.pruning_strategy == 1:
            if len(self.voting_weights) == len(self.models):
                return int(np.argmin(self.voting_weights))
            return 0

        if self.pruning_strategy == 2:
            if base_predictions is not None and y is not None:
                accuracies = np.mean(
                    base_predictions == np.asarray(y, dtype=object),
                    axis=1,
                )
                return int(np.argmin(accuracies))
            return 0

        return 0

    def _remove_model(self, index):
        del self.models[index]
        del self.beta_history[index]
        if len(self.voting_weights) > index:
            del self.voting_weights[index]

    def prune_to_max_size(self, X=None, y=None):
        """Aplica pruning hasta cumplir max_size usando datos etiquetados opcionales."""
        base_predictions = None
        y_array = None

        if X is not None and y is not None and len(self.models) > 0:
            X_records = X.to_dict(orient="records")
            y_array = np.asarray(y, dtype=object)
            base_predictions = self._predict_base_matrix_from_records(X_records)

        self._refresh_voting_weights()

        while len(self.models) > self.max_size:
            remove_index = self._choose_pruning_index(base_predictions, y_array)
            self._remove_model(remove_index)
            if base_predictions is not None:
                base_predictions = np.delete(base_predictions, remove_index, axis=0)

        if self.models:
            self._refresh_voting_weights()

    # -------------------------------------------------
    # Entrenamiento por chunk
    # -------------------------------------------------
    def fit_chunk(self, X, y):
        X_records = X.to_dict(orient="records")
        y_array = np.asarray(y, dtype=object)

        # ---------------------------------------------
        # 1. Evaluar ensemble actual (E_t)
        # ---------------------------------------------
        if self.models:
            old_base_predictions = self._predict_base_matrix_from_records(X_records)
            ensemble_predictions = self._weighted_vote_from_matrix(old_base_predictions)
            errors = ensemble_predictions != y_array
            E_t = float(np.mean(errors))
        else:
            old_base_predictions = np.empty((0, len(X_records)), dtype=object)
            ensemble_predictions = None
            E_t = 0.5  # inicialización razonable

        # ---------------------------------------------
        # 2. Construir distribución D_t
        # ---------------------------------------------
        if ensemble_predictions is None:
            weights = np.ones(len(y_array), dtype=float)
        else:
            weights = np.where(ensemble_predictions == y_array, E_t, 1.0).astype(float)

        D_t = weights / (weights.sum() + 1e-10)

      # ---------------------------------------------
        # 3. Entrenar nuevo clasificador
        # ---------------------------------------------
        new_model = tree.HoeffdingTreeClassifier(
            grace_period=self.grace_period,
            delta=self.delta,
        )

        for xi, yi in zip(X_records, y_array):
            new_model.learn_one(xi, yi)

        self.models.append(new_model)
        self.beta_history.append([])

        # Reutilizamos las predicciones de los modelos antiguos y solo calculamos
        # las del clasificador recién entrenado.
        new_model_predictions = np.array(
            [new_model.predict_one(xi) for xi in X_records],
            dtype=object,
        ).reshape(1, -1)
        all_base_predictions = np.vstack([old_base_predictions, new_model_predictions])

        if self.classes_ is None:
            self.classes_ = np.unique(y_array)

        # ---------------------------------------------
        # 4. Evaluar TODOS los modelos con D_t
        # ---------------------------------------------
        for k, model_predictions in enumerate(all_base_predictions):
            incorrect = model_predictions != y_array
            error = float(np.dot(D_t, incorrect))

            # clipping como en el paper
            error = max(min(error, 0.5), 1e-6)

            beta = error / (1 - error + 1e-10)

            self.beta_history[k].append(beta)

        # ---------------------------------------------
        # 5. Voting weights
        # ---------------------------------------------
        self._refresh_voting_weights()

        while len(self.models) > self.max_size:
            remove_index = self._choose_pruning_index(all_base_predictions, y_array)
            self._remove_model(remove_index)
            all_base_predictions = np.delete(all_base_predictions, remove_index, axis=0)

        if self.models:
            self._refresh_voting_weights()

    # -------------------------------------------------
    # Predicción
    # -------------------------------------------------
    def predict_with_base_predictions(self, X):
    
        if not self.models:
            raise ValueError("El ensemble está vacío.")

        X_records = X.to_dict(orient="records")
        base_predictions = self._predict_base_matrix_from_records(X_records)
        predictions = self._weighted_vote_from_matrix(base_predictions)

        return predictions, base_predictions

    def predict(self, X):
        predictions, _ = self.predict_with_base_predictions(X)
        return predictions