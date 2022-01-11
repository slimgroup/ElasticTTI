import numpy as np
import sympy
from sympy import symbols, Matrix, sqrt, cos, sin

from devito import TensorFunction

__all__ = ['SeismicStiffness', 'Stiffness']


def zero_row(size):
    return Matrix([[0 for _ in range(size)]])


def zero_col(size):
    return Matrix([0 for _ in range(size)])


class Stiffness(Matrix):
    """
    Elastic stiffness tensor with voigt transformation:
    - as_tensor for 6x6 matrix to 3x3x3x3 tensor
    - to_matrix for 3x3x3x3 tensor to 6x6 matrix

    Assumes elastic symmetry (21 coefficients)

    Implements tensor rotation for TTI
    """
    # Voigt notation inidces map
    _voigt = {(0, 0): 0,
              (1, 1): 1,
              (2, 2): 2,
              (1, 2): 3,
              (2, 1): 3,
              (0, 2): 4,
              (2, 0): 4,
              (0, 1): 5,
              (1, 0): 5}

    # Size of the voigt elasticity matrix
    _size = {2: 3, 3: 6}
    _3d = [1, 3, 5]
    _Cij = {'C%s%s' % (i, j): symbols('C%s%s' % (i, j)) for i in range(1, 7) for j in range(i, 7)}

    def __new__(cls, ndim, **kwargs):
        """
        New elasticity matrix with all coefficients
        """
        cls._ndim = ndim
        comps = kwargs.get('comps', cls._init(ndim))
        obj = Matrix.__new__(cls, cls._size[ndim], cls._size[ndim], comps)
        return obj

    def __call__(self, *args, **kwargs):
        return self.subs(kwargs)

    @classmethod
    def cij(cls, i, j):
        ii, jj = np.min((i, j)), np.max((i, j))
        return cls._Cij['C%s%s'% (ii, jj)]

    @classmethod
    def _init(cls, ndim):
        """
        Init elasticity matrix depending on number of dimensions
        """
        if ndim == 3:
            # 21 elastic components
            Cij = [[cls.cij(i, j) for i in range(1, 7)] for j in range(1, 7)]
        else:
            # 21 elastic components
            Cij = [[cls.cij(i, j) for i in cls._3d] for j in cls._3d]
        return sum(Cij, [])

    def rotate(self, R):
        """
        Rotate elasticity matrix with rotation matrix R
        Transforms to tensor then rotate and turn back to matrix
        """
        C = self.as_tensor
        rotated = sympy.tensor.array.MutableDenseNDimArray.zeros(3, 3, 3, 3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for ii in range(3):
                            for jj in range(3):
                                for kk in range(3):
                                    for ll in range(3):
                                        gg = R[ii, i]*R[jj,j]*R[kk, k]*R[ll, l]
                                        rotated[i,j,k,l] += gg*C[ii,jj,kk,ll]
        return self.tensor_to_matrix(sympy.simplify(rotated))

    def prod(self, tau):
        """
        Product of the stiffness tensor with a 2nd order tensor tau
        C : tau [i,k] = sum_kl C_ijkl * Tau_kl
        """
        # Init output
        out = sympy.Matrix(3, 3, lambda i, j: 0)
        # Add zero rows/columns if 2d
        if tau.shape == (2, 2):
            aux = sympy.Matrix(3, 3, lambda i, j: 0)
            aux[0, 0] = tau[0, 0]
            aux[0, 2] = tau[0, 1]
            aux[2, 0] = tau[1, 0]
            aux[2, 2] = tau[1, 1]
        else:
            aux = tau
        # Get stiffness tensor
        tensor = self.as_tensor
        # Compute product
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        out[i, j] += tensor[i,j,k,l] * aux[k, l]

        # Remove extra row/columns if 2D
        if self._ndim == 2:
            final = sympy.Matrix(2, 2, lambda i, j: 0)
            final[0, 0] = out[0, 0]
            final[0, 1] = out[0, 2]
            final[1, 0] = out[2, 0]
            final[1, 1] = out[2, 2]
        else:
            final = out
        # Turn back to Devito type
        return tau.new_from_mat(final)

    @property
    def as_tensor(self):
        """
        Stiffness tensor
        """
        tensor = sympy.tensor.array.MutableDenseNDimArray.zeros(3, 3, 3, 3)
        matrix = self.full_matrix
        for k1, v1 in self._voigt.items():
              i, j = k1
              alpha = v1
              for k2, v2 in self._voigt.items():
                  k, l = k2
                  beta = v2
                  tensor[i, j, k, l] = matrix[alpha, beta]
        return tensor

    @property
    def full_matrix(self):
        """
        Full elasticity matrix with zeros rows/columns if 2D
        """
        if self._ndim == 3:
            return self
        matrix = self
        for i in self._3d:
            matrix = matrix.row_insert(i, zero_row(3))
        for i in self._3d:
            matrix = matrix.col_insert(i, zero_col(6))
        return matrix

    def reduced_matrix(self, matrix):
        """
        Reduce a given elasticity matrix removing zeros row/col in 2D
        """
        if self._ndim == 3:
            return matrix
        for i in self._3d[::-1]:
            matrix.row_del(i)
        for i in self._3d[::-1]:
            matrix.col_del(i)

        return matrix

    def tensor_to_matrix(self, tensor):
        """
        Convert the elasticity matrix to the stiffness tensor via voigt indices
        """
        matrix = self.full_matrix
        matrix.fill(0.)
        for k, v in self._voigt.items():
          i, j = k
          alpha = v
          for k2, v2 in self._voigt.items():
              k, l = k2
              beta = v2
              matrix[alpha, beta] = tensor[i, j, k, l]
        return self.reduced_matrix(matrix)


class SeismicStiffness(Stiffness):
    """
    Stiffness tensor for seismic applications
    """

    @classmethod
    def cij(cls, i, j):
        ii, jj = np.min((i, j)), np.max((i, j))
        if ii == jj or (ii < 4 and jj < 4):
            return cls._Cij['C%s%s'% (ii, jj)]
        else:
            return 0

    def __new__(cls, ndim, vp, vs, **kwargs):
         _subs = cls.process_seismic_kwargs(vp=vp, vs=vs, **kwargs)
         new_obj = Stiffness.__new__(cls, ndim, **kwargs)(**_subs)
         new_obj = new_obj.rotate_tti(new_obj, **kwargs)
         return new_obj(**_subs)

    @classmethod
    def rotate_tti(cls, stiff, **kwargs):
        if 'theta' in kwargs.keys():
            c = cos(kwargs.get('theta'))
            s = sin(kwargs.get('theta'))
            R = Matrix(3, 3, sum([[c, 0, s], [0, 1, 0], [-s, 0, c]], []))
            stiff = stiff.rotate(R)
        if 'phi' in kwargs.keys():
            c = cos(kwargs.get('phi'))
            s = sin(kwargs.get('phi'))
            R = Matrix(3, 3, sum([[1, 0, 0], [0, c, -s], [0, s, c]], []))
            stiff =  stiff.rotate(R)
        return stiff

    @classmethod
    def process_seismic_kwargs(cls, vp=1.5, vs=.75, **kwargs):
        """
        VTI stiffness
        C11   C12   C13   0     0     0
        C12   C22   C23   0     0     0
        C13   C23   C33   0     0     0
        0     0     0     C44   0     0
        0     0     0     0     C55   0
        0     0     0     0     0     C66


        with:

        epsilon = (C11 - C33) / (2 C33)
        gamma = (C66 - C44) / (2 C44)
        delta = .5 * ((C13 + C44)**2 - (C33 - C44)**2)/(C33 * (C33 - C44))
        eta = (epsilon - delta) / (1 + 2 delta)

        leading to:
    
        C33 = rho * vp**2
        C44 = rho * vs**2
        C11 = (2 * epsilon + 1) * C33
        C66 = (2 * epsilon + 1) * C44 (assume gamma = epsilon for now)
        C12 = C11 - 2 C66
        C13 = -C44 + sqrt((C33 - C44)*((2.0*delta + 1) *  C33 - C44))
        C22 = C11
        C23 = C13
        C55 = C44
    
        TTI stiffness is the VTI stiffness rotated with theta and phi

        """
        C33 = vp**2 * kwargs.get('rho', 1)
        C44 = vs**2 * kwargs.get('rho', 1)
        C11 = C33 * (2 * kwargs.get('epsilon', 0) + 1)
        C66 = C44 * (2 * kwargs.get('epsilon', 0) + 1)
        subs = {}
        subs.update({'C33': C33})
        subs.update({'C55': C44})
        subs.update({'C11': C11})
        subs.update({'C22': C11})
        subs.update({'C44': C44})
        subs.update({'C13': sqrt((C33 - C44)*(2.0*C33*kwargs.get('delta', 0) + C33 - C44)) - C44})
        subs.update({'C12': C11 - 2 * C66})
        subs.update({'C22': C11})
        subs.update({'C23': sqrt((C33 - C44)*(2.0*C33*kwargs.get('delta', 0) + C33 - C44)) - C44})
        subs.update({'C66': C66})

        return subs
