from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    from cyanure import BinaryClassifier


class Solver(BaseSolver):
    name = 'Cyanure'

    install_cmd = 'conda'
    requirements = ['pip:cyanure-mkl']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        self.solver = BinaryClassifier(loss='logistic', penalty='l1',
                                       fit_intercept=False)
        self.solver_parameter = dict(
            lambd=self.lmbd / self.X.shape[0], solver='auto', it0=1000000,
            tol=1e-12, verbose=False
        )

    def run(self, n_iter):
        self.solver.fit(self.X, self.y, max_epochs=n_iter,
                        **self.solver_parameter)

    def get_result(self):
        return self.solver.get_weights()
