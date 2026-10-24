import os
import joblib
import numpy as np


class MLModelService:
    def __init__(self):
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../ml/artifacts/iris_model.joblib")
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Artefato do modelo não encontrado em: {model_path}")

        artifact = joblib.load(model_path)
        self.pipeline = artifact['pipeline']
        self.target_names = artifact['target_names']
        self.feature_names = artifact['feature_names']

    def predict(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
        features = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]])

        # Predição e probabilidades
        pred_idx = self.pipeline.predict(features)[0]
        probas = self.pipeline.predict_proba(features)[0]

        predicted_species = self.target_names[pred_idx]
        confidence = float(probas[pred_idx])

        probabilities = {
            name: round(float(prob), 4)
            for name, prob in zip(self.target_names, probas)
        }

        return predicted_species, round(confidence, 4), probabilities


ml_service = MLModelService()
