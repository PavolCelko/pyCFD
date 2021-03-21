import numpy
from matplotlib import pyplot
import time

class Pipeline:
	def __init__(self, dt, duration, x, y, np=50, rho=858, nu=50e-6, p_inlet=None, u_inlet=None, v_inlet=None):
		self.rho = rho
		self.nu = nu
		self.dt = dt
		self.nt = int(duration / dt)
		self.np_iter = np

		self.xx = numpy.array([x for i in range(len(y))])
		self.yy = numpy.array([y for i in range(len(x))]).transpose()

		self.dx = self.xx[1:-1, 1:-1] - self.xx[1:-1, 0:-2]
		self.dy = self.yy[1:-1, 1:-1] - self.yy[0:-2, 1:-1]

		self.u = numpy.zeros(self.xx.shape)
		self.v = numpy.zeros(self.xx.shape)
		self.p = numpy.zeros(self.xx.shape)
		self.b = numpy.zeros(self.xx.shape)
		# self.b = numpy.zeros((ny, nx))

		self.p_inlet = p_inlet
		self.u_inlet = u_inlet
		self.v_inlet = v_inlet

	def build_up_b(self):
		self.b[1:-1, 1:-1] = (self.rho * ( - ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (self.xx[1:-1, 2:] - self.xx[1:-1, 0:-2])) ** 2 -
								2 * ((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (self.yy[2:, 1:-1] - self.yy[0:-2, 1:-1]) *
									 (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (self.xx[1:-1, 2:] - self.xx[1:-1, 0:-2])) -
								((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (self.yy[2:, 1:-1] - self.yy[0:-2, 1:-1])) ** 2))

		return self.b

	def pressure_poisson(self):
		# dx = self.xx[1:-1, 1:-1] - self.xx[1:-1, 0:-2]
		# dy = self.yy[1:-1, 1:-1] - self.yy[0:-2, 1:-1]

		for i in range(self.np_iter):
			pn = self.p.copy()
			self.p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy ** 2 +
							  (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx ** 2) /
							 (2 * (self.dx ** 2 + self.dy ** 2)) -
							 self.dx ** 2 * self.dy ** 2 / (2 * (self.dx ** 2 + self.dy ** 2)) *
							 self.b[1:-1, 1:-1])

			# wall BCs
			self.p[0, :] = self.p[1, :]    # dp/dy = 0 at down wall
			self.p[-1, :] = self.p[-2, :]  # dp/dy = 0 at upper wall

			# inlet BC
			self.p[:, 0] = self.p_inlet   # p = MAX at inlet
			# these are not ordinary BCs, just the inlet pressure is out of area of calculation
			# p[:, 0] = p[:, 1]  # p at inlet
			# outlet BC
			self.p[:, -1] = 0  # p = 0 at outlet

		return self.p

	# def cavity_flow(self, nt, dt, xx, yy, u, v, p, rho, nu):
	def cavity_flow(self):
		nt = self.nt
		dt = self.dt
		xx = self.xx
		yy = self.yy
		u = self.u
		v = self.v
		p = self.p
		rho = self.rho
		nu = self.nu

		for n in range(nt):
			un = u.copy()
			vn = v.copy()

			b_field = self.build_up_b()
			# p = self.pressure_poisson(xx, yy, p, b_field)
			p = self.pressure_poisson()

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

		return u, v, p, self.xx, self.yy


def main():
	# geometry and scaling
	pipe_len   = 5 * 1e-3
	pipe_width = 0.7 * 1e-3
	nx = int(pipe_len * 20 * 1e3) + 1
	ny = int(pipe_width * 20 * 1e3) + 1
	x = numpy.linspace(0, pipe_len, nx)
	y = numpy.linspace(0, pipe_width, ny)
	X, Y = numpy.meshgrid(x, y)
	# xx = numpy.array([x for i in range(len(y))])
	# yy = numpy.array([y for i in range(len(x))]).transpose()

	# fluid physics constants
	rho = 868
	nu = 50e-6

	# timings
	dt = .001/10000
	durat = 0.001
	start_time = time.time()

	# pressure iterator
	npit = 50
	p_inlet = 15e5

	pipe = Pipeline(dt=dt, duration=durat, p_inlet=p_inlet, x=x, y=y, np=npit, rho=858, nu=50e-6)

	u, v, p, xx, yy = pipe.cavity_flow()

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


if __name__ == '__main__':
	main()
