from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        "dataset": ["news20.binary", "rcv1.binary", "SUSY"],
    }

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        if self.dataset == "SUSY":
            self.y = 2 * (self.y > 0) - 1

        data = dict(X=self.X, y=self.y)

        return data
