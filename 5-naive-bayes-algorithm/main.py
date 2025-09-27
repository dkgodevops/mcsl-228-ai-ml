"""
Naive Bayes (Gaussian) classifier - from-scratch implementation + sklearn comparison
Dataset: Iris (4 features, 3 classes) - loaded from sklearn.datasets

How it works (Gaussian NB):
- For each class, compute the mean and variance of each feature (assuming feature-wise
  conditional independence and Gaussian-distributed features).
- For a test sample, compute the Gaussian likelihood for each feature given the class.
- Multiply likelihoods across features, multiply by class prior, choose class with max posterior.
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

# -------------------------
# From-scratch Gaussian NB
# -------------------------
class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = {c: counts[i] / len(y) for i, c in enumerate(self.classes)}
        # For each class, compute mean and var per feature
        self.mean = {}
        self.var = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            # Use unbiased estimator (ddof=1) for variance; add small epsilon to avoid div-by-zero
            self.var[c] = X_c.var(axis=0) + 1e-9

    def _gaussian_pdf(self, x, mean, var):
        # Probability density of x for Gaussian(mean, var) for each feature (vectorized)
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = -((x - mean) ** 2) / (2.0 * var)
        return coeff * np.exp(exponent)

    def predict(self, X):
        y_pred = []
        for x in X:
            class_posteriors = {}
            for c in self.classes:
                # compute likelihood = product of feature-wise pdf values
                pdf_vals = self._gaussian_pdf(x, self.mean[c], self.var[c])
                likelihood = np.prod(pdf_vals)
                posterior = likelihood * self.priors[c]
                class_posteriors[c] = posterior
            # pick the class with highest posterior
            predicted_class = max(class_posteriors.items(), key=lambda item: item[1])[0]
            y_pred.append(predicted_class)
        return np.array(y_pred)

# -------------------------
# Load dataset and run
# -------------------------
def run_demo(test_size=0.3, random_state=42):
    iris = datasets.load_iris()
    X = iris.data      # shape (150, 4)
    y = iris.target    # classes 0,1,2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train from-scratch Gaussian NB
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    y_pred_custom = gnb.predict(X_test)

    # Train sklearn GaussianNB for comparison
    sklearn_gnb = SklearnGaussianNB()
    sklearn_gnb.fit(X_train, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_test)

    # Metrics
    acc_custom = accuracy_score(y_test, y_pred_custom)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print("=== From-scratch Gaussian NB ===")
    print(f"Accuracy: {acc_custom:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_custom, target_names=iris.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_custom))
    print("\n=== sklearn GaussianNB (for comparison) ===")
    print(f"Accuracy: {acc_sklearn:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_sklearn))

if __name__ == "__main__":
    run_demo()
