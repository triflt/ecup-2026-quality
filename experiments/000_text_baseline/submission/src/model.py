import hashlib
import re
from dataclasses import dataclass

import joblib
import numpy as np


_SPACE = re.compile(r"\s+")


def normalize(value):
    return _SPACE.sub(" ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name, description):
    name = normalize(name)
    description = normalize(description)
    return f"{name}\n{name}\n{description}"


def fingerprint(text):
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


@dataclass
class StrongTextBundle:
    vectorizer: object
    category_models: dict
    thresholds: dict
    exact_lookup: dict
    name_lookup: dict

    def predict(self, names, texts, categories):
        matrix = self.vectorizer.transform(texts)
        categories = np.asarray(categories, dtype=object)
        scores = np.full(len(texts), -1e9, dtype=np.float32)
        predictions = np.zeros(len(texts), dtype=np.int8)
        for category, model in self.category_models.items():
            positions = np.flatnonzero(categories == category)
            if positions.size:
                values = model.decision_function(matrix[positions])
                scores[positions] = values
                predictions[positions] = values >= self.thresholds[category]
        for index, (name, text, category) in enumerate(zip(names, texts, categories)):
            exact_key = (str(category), fingerprint(text))
            name_key = (str(category), normalize(name))
            if exact_key in self.exact_lookup:
                predictions[index] = self.exact_lookup[exact_key]
            elif name_key in self.name_lookup:
                predictions[index] = self.name_lookup[name_key]
        return scores, predictions

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)
