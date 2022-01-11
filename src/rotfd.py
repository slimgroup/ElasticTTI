
from sympy import sqrt, finite_diff_weights, Rational
from sympy.physics.units import * 

from devito.finite_differences.differentiable import Add
from devito.types.tensor import vec_func, tens_func

__all__ = ['dx45', 'dz45', 'dxrot', 'dzrot', 'divrot', 'gradrot']

def dz45(field, x0=0.5):
    x, z = field.grid.dimensions
    ix, iz = field.indices[1:]
    r = sqrt(x.spacing**2 + z.spacing**2)
    o = field.space_order

    coeffs = finite_diff_weights(1, [i for i in range(- o//2 + 1, o//2 + 1)], .5)[-1][-1]

    if x0 == 0:
        inds =  [i for i in range(- o//2 + 1, o//2 + 1)]
    # Deriv of stagg (x, y) at x
    else:
        inds = [Rational(1,2) + i for i in range(- o//2, o//2)]
    termsdz = []
    for i, c in zip(inds, coeffs):
        ztild = field.subs({ix: x + i * x.spacing, iz: z + i * z.spacing})
        termsdz.append(ztild * c / r)
    return  Add(*termsdz)

def dx45(field, x0=0.5):
    x, z = field.grid.dimensions
    ix, iz = field.indices[1:]
    r = sqrt(x.spacing**2 + z.spacing**2)
    o = field.space_order
    coeffs = finite_diff_weights(1, [i for i in range(- o//2 + 1, o//2 + 1)], .5)[-1][-1]

    if x0 == 0:
        inds =  [i for i in range(- o//2 + 1, o//2 + 1)]
    # Deriv of stagg (x, y) at x
    else:
        inds = [i +  Rational(1,2) for i in range(- o//2, o//2)]
    termsdx = []
    for i, c, i2 in zip(inds, coeffs, inds[::-1]):
        xtild = field.xreplace({ix: x + i * x.spacing, iz: z + i2 * z.spacing})
        termsdx.append(xtild * c / r)

    return Add(*termsdx)


def dxrot(field, x0=0.5):
    x, z = field.grid.dimensions
    r = sqrt(x.spacing**2 + z.spacing**2)
    return .5 * field.dx + .5 * r / (2 * x.spacing) * (dx45(field, x0=x0) + dz45(field, x0=x0))

def dzrot(field, x0=0.5):
    x, z = field.grid.dimensions
    r = sqrt(x.spacing**2 + z.spacing**2)
    return .5 * field.dy + .5 * r / (2 * z.spacing) * (-dx45(field, x0=x0) + dz45(field, x0=x0))

dif = {'dx': dxrot, 'dy': dzrot}


def divrot(self):
    comps = []
    to = getattr(self, 'time_order', 0)
    func = vec_func(self, self)
    for j, d in enumerate(self.space_dimensions):
        comps.append(sum([dif['d%s' % d.name](self[j, i], x0=0)
                         for i, d in enumerate(self.space_dimensions)]))
    return func(name='div_%s' % self.name, grid=self[0].grid,
                space_order=self.space_order, components=comps, time_order=to)

def gradrot(self):
        to = getattr(self, 'time_order', 0)
        func = tens_func(self, self)
        comps = [[dif['d%s' % d.name](f, x0=0.5) for d in self.space_dimensions]
                 for f in self]
        return func(name='grad_%s' % self.name, grid=self[0].grid, time_order=to,
                    space_order=self.space_order, components=comps, symmetric=False)