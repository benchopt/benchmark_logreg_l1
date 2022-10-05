from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from skglm.estimators import SparseLogisticRegression
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/skglm.git@main'
    ]
    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.logreg = SparseLogisticRegression(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000,
            tol=1e-12, fit_intercept=False, warm_start=False)

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.logreg.max_iter = n_iter
            self.logreg.fit(self.X, self.y)

            coef = self.logreg.coef_.flatten()
            self.coef = coef

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.coef
