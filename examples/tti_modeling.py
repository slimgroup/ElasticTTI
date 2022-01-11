import numpy as np
import time
import matplotlib.pyplot as plt
from src import SeismicStiffness

from devito import Function, VectorTimeFunction, TensorTimeFunction, Grid, Eq, Operator, TensorFunction, NODE
from examples.seismic import RickerSource, TimeAxis
from examples.seismic.model import initialize_damp

from src import divrot, gradrot

grid = Grid((481, 481), extent=(2400., 2400.), origin=(-200, -200), dtype=np.float32)
x, z = grid.dimensions

# VTI parama
vp = Function(name="vp", grid=grid, space_order=0)
vs = Function(name="vs", grid=grid, space_order=0)
epsilon = Function(name="epsilon", grid=grid, space_order=0)
delta = Function(name="delta", grid=grid, space_order=0)
theta = Function(name="theta", grid=grid, space_order=0)
phi = Function(name="phi", grid=grid, space_order=0)
rho = Function(name="rho", grid=grid, space_order=0)
damp = Function(name="damp", grid=grid, space_order=0)
initialize_damp(damp, ((40, 40), (40, 40)), grid.spacing, abc_type="mask")

vp.data.fill(1.5)
vp.data[:, 101:] = 2.4
vs.data[:] =  vp.data[:] - 1.5
rho.data[:] = 1.0
epsilon.data[:] = (vp.data[:] - 1.5)/3.
delta.data[:] = (vp.data[:] - 1.5)/5.
theta.data[:] = vp.data[:] - 1.5
phi.data[:] = np.pi/20

t0 = time.time()
C = SeismicStiffness(grid.dim, vp, vs=vs, rho=rho, epsilon=epsilon, delta=delta, theta=theta)
print("Init C took %s" % (time.time() -t0,))

# Modeling
v = VectorTimeFunction(name="v", grid=grid, time_order=1, space_order=8, staggered=((x, z), (x, z)))
tau = TensorTimeFunction(name="t", grid=grid, space_order=8, time_order=1, staggered=((NODE, NODE), (NODE, NODE)))

t0, tn = 0., 750.
f0 = 0.020
dt = .6 * grid.spacing[0] / (np.sqrt(3) * np.max(vp.data) * np.sqrt(1 + 2 *  np.max(epsilon.data)))
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=f0, time_range=time_range)
src.coordinates.data[0, :] = [1000., 20.]

src_term_v = []
src_term_t = []
src_term_t += src.inject(field=tau.forward[0, 0], expr=src)
src_term_t += src.inject(field=tau.forward[1, 1], expr=src)

# Equation
u_v = Eq(v.forward, damp*(v + dt/rho*divrot(tau)))

e = 1 / 2 * (gradrot(v.forward) + gradrot(v.forward).T)
u_t = Eq(tau.forward, damp*(tau + dt * C.prod(e)))

op = Operator([u_v] + src_term_v + [u_t] + src_term_t, subs=grid.spacing_map)
op()


scale = 1e-2

plt.figure(figsize=(15,15))
plt.subplot(221)
plt.imshow(v[0].data[0].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vx")
plt.subplot(222)
plt.imshow(v[1].data[0].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vz")
plt.subplot(223)
plt.imshow(tau[0, 0].data[0].T + tau[1, 1].data[0].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("P (trace(tau))")
plt.subplot(224)
plt.imshow(tau[0, 1].data[0].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("txz")
plt.tight_layout()

plt.show()