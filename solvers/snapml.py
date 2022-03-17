from benchopt import BaseSolver, safe_import_context
import warnings

with safe_import_context() as import_ctx:
    # from snapml import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "solver": [
            # 'saga',
            "liblinear"
        ],
    }
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.clf = LogisticRegression(
            solver=self.solver,
            C=1 / self.lmbd,
            penalty="l1",
            fit_intercept=False,
            tol=1e-12,
        )

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
