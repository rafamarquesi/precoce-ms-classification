from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier


class ClfSwitcher(BaseEstimator):

    def __init__(
        self,
        estimator: object = SGDClassifier()
    ):
        """A custom BaseEstimator that can switch between classifiers.

        Args:
            estimator (object): Sklearn object - The classifier.
        """

        self.estimator = estimator

    def fit(self, x, y=None, **kwargs):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def score(self, x, y):
        return self.estimator.score(x, y)
