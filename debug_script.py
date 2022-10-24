import numpy as np

from skglm.datafits import Logistic
from skglm.penalties import L1
from skglm.solvers import ProxNewton
from skglm.estimators import GeneralizedLinearEstimator

from benchopt.datasets import make_correlated_data


n_samples, n_features = 500, 5000
rho = 0.5

X, y, _ = make_correlated_data(n_samples, n_features,
                               random_state=123)
y = np.sign(y)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = rho * alpha_max

estimator = GeneralizedLinearEstimator(
    datafit=Logistic(),
    penalty=L1(alpha),
    solver=ProxNewton(tol=1e-12, fit_intercept=False, verbose=2)
)

estimator.fit(X, y)
