import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy as np
from scipy.io import mmread

# Base class for preconditioners. The implementation of the base class is
# a do-nothing two-sided preconditioner. Derived classes should reimplement
# at least one of the applyLeft() and applyRight() functions.
class PreconditionerBase:
    # Nothing for constructor to do
    def __init__(self):
        pass


    def applyLeft(self, vec):
        return vec

    def applyRight(self, vec):
        return vec




# ILU Preconditioner, applied from the right.
# Constructor parameters:
# (*) A           -- The matrix to be approximately factored.
# (*) drop_tol    -- Tolerance for deciding whether to accept "fill" entries
#                  in the approximate factorization. With a value of 1, no fill
#                  is accepted. With a value of 0, all fill is accepted. Default
#                  is 0.0001.
# (*) fill_factor -- Factor by which the number of nonzeros is allowed
#                  to increase during the approximate factorization. SuperLU's
#                  default is 10. With a low value of drop_tol you may need to
#                  increase this.

class ILURightPreconditioner(PreconditionerBase):
    def __init__(self, A, drop_tol=0.001, fill_factor=15):
        PreconditionerBase.__init__(self)

        # Construct the ILU approximate factorization. Super LU needs
        # to use CSC format, so do the conversion here.
        self.ILU = spla.spilu(A.tocsc(),
            drop_tol=drop_tol, fill_factor=fill_factor)

    def residCheck(self):
        L = self.ILU.L
        U = self.ILU.U
        pr = self.ILU.perm_r
        pc = self.ILU.perm_c
        n,nc = U.shape
        Pr = sp.csc_matrix((np.ones(n), (pr, np.arange(n))))
        Pc = sp.csc_matrix((np.ones(n), (np.arange(n), pc)))
        resid = L*U - Pr*A*Pc
        normOrder=1 # easiest to use 1 norm with CSC storage.
        normA = spla.norm(A,normOrder)
        return spla.norm(resid, normOrder)/normA


    def applyRight(self, vec):
        return self.ILU.solve(vec)


if __name__=='__main__':

    refinement_level = 12
    A = mmread('DH-Matrix-%d.mtx' % refinement_level)
    A = A.tocsr()
    n,nc = A.shape

    fill_factor = 10

    print('----------------------------------------------------------------------')
    print('Checking ILU factorization by computing ||A-LU||/||A|| (with permutations)')
    print('System is %d by %d' %(n,nc))
    print('ILU maximum fill factor is %f' % fill_factor)

    print('\n\n%20s\t %20s' % ('Drop tolerance', 'ILU residual norm'))
    print('----------------------------------------------------------------------')
    for p in range(20):
        tol = 0.5**p
        ILU = ILURightPreconditioner(A, drop_tol=tol, fill_factor=20)
        print('%20.15f\t %20.5g'%(tol, ILU.residCheck()))
