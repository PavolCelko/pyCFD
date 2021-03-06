import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


def build_up_b(b, rho, dt, u, v, dx, dy):
	b[1:-1, 1:-1] = (rho * ( - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
							2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
								 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
							((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

	return b


def pressure_poisson(p, dx, dy, b):
	pn = numpy.empty_like(p)
	pn = p.copy()

	for q in range(nit):
		pn = p.copy()
		p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
						  (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
						 (2 * (dx ** 2 + dy ** 2)) -
						 dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
						 b[1:-1, 1:-1])

		# wall BCs
		p[0, :] = p[1, :]    # dp/dy = 0 at down wall
		p[-1, :] = p[-2, :]  # dp/dy = 0 at upper wall

		# inlet BC
		# these are not ordinary BCs, just the inlet pressure is out of area of calculation
		p[:, 0] = p[:, 1]  # p at inlet
		# p[:, 0] = 10   # p = MAX at inlet
		# outlet BC
		p[:, -1] = 0  # p = 0 at outlet

	return p


##variable declarations
pipe_len   = 10
pipe_width = 2

nx = pipe_len * 20 + 1
ny = pipe_width * 20 + 1
# nt = 10
nit = 50
c = 1
dx = pipe_len / (nx - 1)
dy = pipe_width / (ny - 1)
x = numpy.linspace(0, pipe_len, nx)
y = numpy.linspace(0, pipe_width, ny)
X, Y = numpy.meshgrid(x, y)


##physical variables
rho = 1
nu = .1
dt = .001/2
durat = 1
u_inlet = 10

#initial conditions
u = numpy.zeros((ny, nx))
un = numpy.zeros((ny, nx))

v = numpy.zeros((ny, nx))
vn = numpy.zeros((ny, nx))

p = numpy.ones((ny, nx))
pn = numpy.ones((ny, nx))

b = numpy.zeros((ny, nx))


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
	un = numpy.empty_like(u)
	vn = numpy.empty_like(v)
	b = numpy.zeros((ny, nx))

	for n in range(nt):
		# wall BC - no slip on the walls
		# u[0, :] = 0
		# v[0, :] = 0
		# u[-1, :] = 0
		# v[-1, :] = 0
		#
		# u[1:-1, 0] = u_inlet # BC u at x = 0
		# v[1:-1, 0] = 0  # BC v at x = 0
		# u[1:-1, -1] = u[1:-1, -2] # BC u at x = 0

		un = u.copy()
		vn = v.copy()

		b = build_up_b(b, rho, dt, u, v, dx, dy)
		p = pressure_poisson(p, dx, dy, b)

		u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
						 un[1:-1, 1:-1] * dt / dx *
						 (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
						 vn[1:-1, 1:-1] * dt / dy *
						 (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
						 dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
						 nu * (dt / dx ** 2 *
							   (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
							   dt / dy ** 2 *
							   (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

		v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
						 un[1:-1, 1:-1] * dt / dx *
						 (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
						 vn[1:-1, 1:-1] * dt / dy *
						 (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
						 dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
						 nu * (dt / dx ** 2 *
							   (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
							   dt / dy ** 2 *
							   (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

		# wall BCs - no slip on the walls
		u[0, :] = 0
		v[0, :] = 0
		u[-1, :] = 0
		v[-1, :] = 0

		# inlet BCs
		u[1:-1, 0] = u_inlet  # BC u at inlet
		# u[1:-1, 0] = u[1:-1, 1]  # BC u at inlet
		v[1:-1, 0] = 0        # BC v at inlet

		# these are not ordinary BCs, just the outlet velocity is out of area of calculation
		u[1:-1, -1] = u[1:-1, -2]  # u at outlet layer
		v[1:-1, -1] = v[1:-1, -2]  # v at outlet layer

	return u, v, p


u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
nt = int(durat / dt)
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

# print(u[20, -3:])
# print(p[20, -3:])
print("pressure outlet : {:0.2f}  {:0.2f}  {:0.2f}".format(*tuple(p[20, -3:])))
print("pressure inlet  : {:0.2f}  {:0.2f}  {:0.2f}".format(*tuple(p[20, :3])))
print("velocity outlet : {:0.2f}  {:0.2f}  {:0.2f}".format(*tuple(u[20, -3:])))
print("velocity inlet  : {:0.2f}  {:0.2f}  {:0.2f}".format(*tuple(u[20, :3])))

pyplot.imshow(u, cmap='jet')
pyplot.colorbar()
pyplot.show()

# pyplot.plot(x, u[20, :])
# pyplot.show()
