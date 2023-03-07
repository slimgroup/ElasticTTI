import numpy as np
import time
import matplotlib.pyplot as plt
from src import SeismicStiffness

from devito import Function, VectorTimeFunction, TensorTimeFunction, Grid, Eq, Operator, TensorFunction, NODE, div, grad
from examples.seismic import RickerSource, TimeAxis
from examples.seismic.model import initialize_damp

from src import divrot, gradrot

grid = Grid((241, 241, 241), extent=(2400., 2400., 2400.), origin=(-200, -200, -200), dtype=np.float32)
x, y, z = grid.dimensions
midx, midy = 121, 121

# VTI parama
vp = Function(name="vp", grid=grid, space_order=0)
vs = Function(name="vs", grid=grid, space_order=0)
epsilon = Function(name="epsilon", grid=grid, space_order=0)
delta = Function(name="delta", grid=grid, space_order=0)
rho = Function(name="rho", grid=grid, space_order=0)
damp = Function(name="damp", grid=grid, space_order=0)
initialize_damp(damp, ((20, 20), (20, 20), (20, 20)), grid.spacing, abc_type="mask")

vp.data.fill(1.5)
vp.data[:, :, 101:] = 2.4
vs.data[:] =  vp.data[:] - 1.5
rho.data[:] = 1.0
epsilon.data[:] = (vp.data[:] - 1.5)/3.
delta.data[:] = (vp.data[:] - 1.5)/5.

t0 = time.time()
C = SeismicStiffness(grid.dim, vp, vs=vs, rho=rho, epsilon=epsilon, delta=delta)
print("Init C took %s" % (time.time() -t0,))

# Modeling
v = VectorTimeFunction(name="v", grid=grid, time_order=1, space_order=8)
tau = TensorTimeFunction(name="t", grid=grid, space_order=8, time_order=1)

t0, tn = 0., 1000.
f0 = 0.010
dt = np.float32(.6 * grid.spacing[0] / (np.sqrt(3) * np.max(vp.data) * np.sqrt(1 + 2 *  np.max(epsilon.data))))
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=f0, time_range=time_range)
src.coordinates.data[0, :] = [1000., 1000., 20.]

src_term_v = []
src_term_t = []
src_term_t += src.inject(field=tau.forward[0, 0], expr=src)
src_term_t += src.inject(field=tau.forward[1, 1], expr=src)
src_term_t += src.inject(field=tau.forward[2, 2], expr=src)

# Equation
u_v = Eq(v.forward, damp*(v + dt/rho*div(tau)))

e = 1 / 2 * (grad(v.forward) + grad(v.forward).transpose(inner=False))
u_t = Eq(tau.forward, damp*(tau + dt * C.prod(e)))

op = Operator([u_v] + src_term_v + [u_t] + src_term_t, subs=grid.spacing_map)
op()


scale = 1e-3

fig = plt.figure(figsize=(10,10))
fig.suptitle("Middle x slice", fontsize=14)
plt.subplot(221)
plt.imshow(v[0].data[0][:, midy, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vx")
plt.subplot(222)
plt.imshow(v[2].data[0][:, midy, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vz")
plt.subplot(223)
plt.imshow(tau[0, 0].data[0][:, midy, :].T + tau[1, 1].data[0][:, midy, :].T + tau[2, 2].data[0][:, midy, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("P (trace(tau))")
plt.subplot(224)
plt.imshow(tau[0, 2].data[0][:, midy, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("txz")
plt.tight_layout()
plt.savefig("3DEVTI-x.png", bbox_inches="tight")


fig = plt.figure(figsize=(10,10))
fig.suptitle("Middle y slice", fontsize=14)
plt.subplot(221)
plt.imshow(v[1].data[0][midx, :, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vx")
plt.subplot(222)
plt.imshow(v[2].data[0][midx, :, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("vz")
plt.subplot(223)
plt.imshow(tau[0, 0].data[0][midx, :, :].T + tau[1, 1].data[0][midx, :, :].T +  tau[2, 2].data[0][midx, :, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("P (trace(tau))")
plt.subplot(224)
plt.imshow(tau[1, 2].data[0][midx, :, :].T, vmin=-scale, vmax=scale, cmap="seismic")
plt.title("tyz")
plt.tight_layout()
plt.savefig("3DEVTI-y.png", bbox_inches="tight")


plt.show()

from IPython import embed; embed()