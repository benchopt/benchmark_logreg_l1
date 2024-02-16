from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    from skglm.penalties import L1
    from skglm.datafits import Logistic
    from skglm.solvers import ProxNewton
    from skglm.estimators import GeneralizedLinearEstimator
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:skglm>=0.2',
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
        self.estimator = GeneralizedLinearEstimator(
            datafit=Logistic(),
            penalty=L1(lmbd / n_samples),
            solver=ProxNewton(tol=1e-12, fit_intercept=False)
        )

        # Perform 5 iteration of solver to cache Numba compilation
        # and avoid wiggly objective curves
        self.run(5)

    def run(self, n_iter):
        self.estimator.solver.max_iter = n_iter
        self.estimator.fit(self.X, self.y)
        self.coef = self.estimator.coef_.flatten()

    def get_next(self, stop_val):
        return stop_val + 1

    def get_result(self):
        return dict(beta=self.coef)
