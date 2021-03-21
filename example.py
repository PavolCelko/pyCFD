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

		self.p_inlet = p_inlet
		self.u_inlet = u_inlet
		self.v_inlet = v_inlet

	def build_up_b(self):
		self.b[1:-1, 1:-1] = (self.rho * ( - ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (self.xx[1:-1, 2:] - self.xx[1:-1, 0:-2])) ** 2 -
								2 * ((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (self.yy[2:, 1:-1] - self.yy[0:-2, 1:-1]) *
									 (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (self.xx[1:-1, 2:] - self.xx[1:-1, 0:-2])) -
								((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (self.yy[2:, 1:-1] - self.yy[0:-2, 1:-1])) ** 2))

	def pressure_poisson(self):

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

	def cavity_flow(self):

		for n in range(self.nt):
			un = self.u.copy()
			vn = self.v.copy()

			self.build_up_b()
			self.pressure_poisson()

			self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
							un[1:-1, 1:-1] * self.dt / self.dx *
							(un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
							vn[1:-1, 1:-1] * self.dt / self.dy *
							(un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
							self.dt / (2 * self.rho * self.dx) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
							self.nu * (self.dt / self.dx ** 2 *
									(un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
									self.dt / self.dy ** 2 *
									(un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

			self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
							un[1:-1, 1:-1] * self.dt / self.dx *
							(vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
							vn[1:-1, 1:-1] * self.dt / self.dy *
							(vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
							self.dt / (2 * self.rho * self.dy) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
							self.nu * (self.dt / self.dx ** 2 *
									(vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
									self.dt / self.dy ** 2 *
									(vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

			# wall BCs - no slip on the walls
			self.u[0, :] = 0
			self.v[0, :] = 0
			self.u[-1, :] = 0
			self.v[-1, :] = 0

			# inlet BCs
			self.u[1:-1, 0] = numpy.mean(self.u[1:-1, -1])  # BC u at inlet
			self.v[1:-1, 0] = 0        # BC v at inlet

			# these are not ordinary BCs, just the outlet velocity is out of area of calculation
			self.u[1:-1, -1] = self.u[1:-1, -2]  # u at outlet layer
			self.v[1:-1, -1] = self.v[1:-1, -2]  # v at outlet layer

		return self.u, self.v, self.p, self.xx, self.yy


def main():
	# fluid physics constants
	rho = 868
	nu = 50e-6

	# geometry and scaling
	pipe_len   = 5 * 1e-3
	pipe_width = 0.7 * 1e-3
	nx = int(pipe_len * 20 * 1e3) + 1
	ny = int(pipe_width * 20 * 1e3) + 1
	x = numpy.linspace(0, pipe_len, nx)
	y = numpy.linspace(0, pipe_width, ny)
	X, Y = numpy.meshgrid(x, y)

	# timings
	dt = .001/10000
	dur = 0.003

	# pressure iterator
	np_init = 10
	# p_inlet = 15e5
	numpy.seterr(invalid='raise')
	numpy.seterr(over='raise')

	for p_bars in range(1, 24, 1):
		for np in range(np_init, 210, 10):
			print("np = {:d}".format(np))
			try:
				p_inlet = p_bars * 1e5
				# simulate flow
				start_time = time.time()
				pipe = Pipeline(dt=dt, duration=dur, p_inlet=p_inlet, x=x, y=y, np=np, rho=rho, nu=nu)
				u, v, p, xx, yy = pipe.cavity_flow()
				# (u, v, p, xx, yy) = (pipe.u, pipe.v, pipe.p, pipe.xx, pipe.yy)
				end_time = time.time()
			except FloatingPointError:
				continue
			else:
				np_init = np
				break

		print("script run for {:0.0f} seconds".format(end_time - start_time))
		print("rho={:0.0f} nu={:0.1f} duration={:0.0f}ms		dt={:0.3f}us nx={:d} ny={:d} npit={:d}".format(rho, nu, dur*1e3,		dt*1e6, nx, ny, np))
		print("pressure inlet  : {:0.2f}".format(*tuple(p[int(ny/2), :1])))
		print("mean velocity inlet   : {:0.2f} ".format((numpy.mean(u[1:-1, 0]))))
		print("mean velocity outlet  : {:0.2f} ".format((numpy.mean(u[1:-1, -1]))))
		print("outlet flow : {:0.3f} cm2/ms".format(sum(u[1:-1, -1] * (yy[1:-1, -1] - yy[0:-2, -1])) * 1e4 / 1e3))

		numpy.save("simul_output/x_coord_array"  + str(p_bars), x)
		numpy.save("simul_output/y_coord_array"  + str(p_bars), y)
		numpy.save("simul_output/xx_coord_array" + str(p_bars), xx)
		numpy.save("simul_output/yy_coord_array" + str(p_bars), yy)
		numpy.save("simul_output/pressure_array" + str(p_bars), p)
		numpy.save("simul_output/velocity_array" + str(p_bars), u)

	sum_u = numpy.zeros((1, nx))
	for i in range(nx):
		sum_u[0, i] = numpy.mean(u[1:-1, i])

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
