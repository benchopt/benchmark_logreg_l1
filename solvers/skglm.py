from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from skglm.penalties import L1
    from skglm.datafits import Logistic
    from skglm.solvers import ProxNewton
    from skglm.utils import compiled_clone
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
        self.l1_penalty = compiled_clone(L1(lmbd / n_samples))
        self.log_datafit = compiled_clone(Logistic())
        self.prox_solver = ProxNewton(tol=1e-12, fit_intercept=False)

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        self.prox_solver.max_iter = n_iter
        coef = self.prox_solver.solve(self.X, self.y, self.log_datafit,
                                      self.l1_penalty)[0]
        self.coef = coef.flatten()

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.coef
