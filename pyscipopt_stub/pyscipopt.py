"""
Stub for running learn2branch's 03_train_gcnn.py without a compiled SCIP +
PySCIPOpt install.

learn2branch/utilities.py imports the real pyscipopt at module level, but
03_train_gcnn.py's training path (loading already-generated sample_*.pkl and
training on them) never actually calls into scip.* -- that only happens in
utilities.py's data-generation-time helpers, which 03_train_gcnn.py never
calls. Put this directory first on PYTHONPATH to satisfy the import without
installing the real SCIP + PySCIPOpt toolchain:

    PYTHONPATH=/path/to/pyscipopt_stub python 03_train_gcnn.py setcover -g 0
"""


class Branchrule:
    pass


class SCIP_PARAMSETTING:
    OFF = None
