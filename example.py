import numpy
from matplotlib import pyplot
import time


def build_up_b(xx, yy, b, rho, u, v):
	b[1:-1, 1:-1] = (rho * ( - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (xx[1:-1, 2:] - xx[1:-1, 0:-2])) ** 2 -
							2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (yy[2:, 1:-1] - yy[0:-2, 1:-1]) *
								 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (xx[1:-1, 2:] - xx[1:-1, 0:-2])) -
							((v[2:, 1:-1] - v[0:-2, 1:-1]) / (yy[2:, 1:-1] - yy[0:-2, 1:-1])) ** 2))

	return b


def pressure_poisson(xx, yy, p, b):
	dx = xx[1:-1, 1:-1] - xx[1:-1, 0:-2]
	dy = yy[1:-1, 1:-1] - yy[0:-2, 1:-1]

	for q in range(npit):
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
		p[:, 0] = p_inlet   # p = MAX at inlet
		# these are not ordinary BCs, just the inlet pressure is out of area of calculation
		# p[:, 0] = p[:, 1]  # p at inlet
		# outlet BC
		p[:, -1] = 0  # p = 0 at outlet

	return p


def cavity_flow(nt, dt, xx, yy, u, v, p, rho, nu):
	b_field = numpy.zeros((ny, nx))

	for n in range(nt):
		un = u.copy()
		vn = v.copy()

		b_field = build_up_b(xx, yy, b_field, rho, u, v)
		p = pressure_poisson(xx, yy, p, b_field)

		dx = xx[1:-1, 1:-1] - xx[1:-1, 0:-2]
		dy = yy[1:-1, 1:-1] - yy[0:-2, 1:-1]

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
		u[1:-1, 0] = numpy.mean(u[1:-1, -1])  # BC u at inlet
		v[1:-1, 0] = 0        # BC v at inlet

		# these are not ordinary BCs, just the outlet velocity is out of area of calculation
		u[1:-1, -1] = u[1:-1, -2]  # u at outlet layer
		v[1:-1, -1] = v[1:-1, -2]  # v at outlet layer

	return u, v, p


# geometry and scaling
pipe_len   = 5 * 1e-3
pipe_width = 0.7 * 1e-3
nx = int(pipe_len * 20 * 1e3) + 1
ny = int(pipe_width * 20 * 1e3) + 1
x = numpy.linspace(0, pipe_len, nx)
y = numpy.linspace(0, pipe_width, ny)
X, Y = numpy.meshgrid(x, y)
xx = numpy.array([x for i in range(len(y))])
yy = numpy.array([y for i in range(len(x))]).transpose()

# fluid physics constants
rho = 868
nu = 50e-6

# timings
dt = .001/10000
durat = 0.001
nt = int(durat / dt)
start_time = time.time()

# pressure iterator
npit = 50
p_inlet = 15e5

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))

u, v, p = cavity_flow(nt, dt, xx, yy, u, v, p, rho, nu)

sum_u = numpy.zeros((1, nx))
for i in range(nx):
	sum_u[0, i] = numpy.mean(u[1:-1, i])

end_time = time.time()

print("script run for {:0.0f} seconds".format(end_time - start_time))
print("nu={:0.1f} durat={:0.0f}ms		dt={:0.3f}us nx={:d} ny={:d}".format(nu, durat*1e3,		dt*1e6, nx, ny))
print("pressure inlet  : {:0.2f}".format(*tuple(p[int(ny/2), :1])))
print("mean velocity inlet   : {:0.2f} ".format((numpy.mean(u[1:-1, 0]))))
print("mean velocity outlet  : {:0.2f} ".format((numpy.mean(u[1:-1, -1]))))
print("outlet flow : {:0.3f} cm2/ms".format(sum(u[1:-1, -1] * (yy[1:-1, -1] - yy[0:-2, -1])) * 1e4 / 1e3))

numpy.save("x_coord_array", x)
numpy.save("y_coord_array", y)
numpy.save("xx_coord_array", xx)
numpy.save("yy_coord_array", yy)
numpy.save("pressure_array", p)
numpy.save("velocity_array", u)


pyplot.subplot(4, 1, 1)
pyplot.imshow(u, cmap='jet')
pyplot.colorbar()

pyplot.subplot(4, 1, 2)
pyplot.imshow(p, cmap='jet')
pyplot.colorbar()

pyplot.subplot(4, 1, 3)
pyplot.plot(x, u[int(ny/2), :])

pyplot.subplot(4, 1, 4)
pyplot.plot(x, sum_u[0, :])

pyplot.show()
