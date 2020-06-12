import qutip as qt
import numpy as np

def expand_operator_dim(op, N):
    d = op.shape[0]
    new_op = np.diag([1]*N).astype('complex')
    new_op[:d, :d] = op

    return qt.Qobj(new_op)