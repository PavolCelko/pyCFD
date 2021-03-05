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

		# p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
		# p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
		# p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
		# p[-1, :] = 0  # p = 0 at y = 2

		p[:, -1] = 0  # p = 0 at x = 2
		p[:, 0] = 100   # p = MAX at x = 0
		p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
		p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2

	return p


##variable declarations
pipe_len   = 10
pipe_width = 2

nx = pipe_len * 20 + 1
ny = pipe_width * 20 + 1
nt = 10
nit = 50
c = 1
dx = pipe_len / (nx - 1)
dy = pipe_width / (ny - 1)
x = numpy.linspace(0, pipe_len, nx)
y = numpy.linspace(0, pipe_width, ny)
X, Y = numpy.meshgrid(x, y)


##physical variables
rho = 1
nu = .3
F = 1
dt = .001

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

		# u[0, :] = 0
		# u[:, 0] = 0
		# u[:, -1] = 0
		# u[-1, :] = 1  # set velocity on cavity lid equal to 1
		# v[0, :] = 0
		# v[-1, :] = 0
		# v[:, 0] = 0
		# v[:, -1] = 0

		u[0, :] = 0
		v[0, :] = 0
		u[-1, :] = 0
		v[-1, :] = 0

		u[:, 0] = u[:, 1]
		u[:, -1] = u[:, -2]
		v[:, 0] = v[:, 1]
		v[:, -1] = v[:, -2]

	return u, v, p


u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
nt = 1000
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

pyplot.imshow(u, cmap='jet')
pyplot.colorbar()
pyplot.show()
