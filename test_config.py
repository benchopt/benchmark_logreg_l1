import sys

import pytest


def xfail_test_solver_install(solver_class):

    if solver_class.name.lower() == 'cyanure' and sys.platform == 'darwin':
        pytest.xfail('Cyanure is not easy to install on macos.')

    # Lightning install is broken on python3.9+.
    # See issue #XX
    if (solver_class.name.lower() == 'lightning'
            and sys.version_info >= (3, 9)):
        pytest.xfail('Lightning install is broken on python3.9+.')
