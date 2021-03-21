import numpy
from matplotlib import pyplot


def main():
	plt_pressure = []
	plt_outflow = []
	plt_deviations = []
	print("pressure [bar] | outflow [cm2/ms] | flow deviation [%]")
	for simul_idx in range(1, 24):
		x = numpy.load("simul_output/x_coord_array" + str(simul_idx) + ".npy")
		y = numpy.load("simul_output/y_coord_array" + str(simul_idx) + ".npy")
		xx = numpy.load("simul_output/xx_coord_array" + str(simul_idx) + ".npy")
		yy = numpy.load("simul_output/yy_coord_array" + str(simul_idx) + ".npy")
		p = numpy.load("simul_output/pressure_array" + str(simul_idx) + ".npy")
		u = numpy.load("simul_output/velocity_array" + str(simul_idx) + ".npy")
		# v = numpy.load("simul_output/velocity_v_array" + str(simul_idx) + ".npy")
		nx = len(x)
		ny = len(y)

		sum_u = numpy.zeros((1, nx))
		for i in range(nx):
			sum_u[0, i] = numpy.mean(u[1:-1, i])
		plt_deviations.append((sum_u.max() - sum_u.min()) / sum_u.min() * 100)
		outflow = sum(u[1:-1, -1] * (yy[1:-1, -1] - yy[0:-2, -1])) * 1e4 / 1e3
		plt_pressure.append(p[4, 0]/1e5)
		plt_outflow.append(outflow)
		print("       {:0.0f} |      {:0.3f}     | {:0.1f}".format(plt_pressure[-1], plt_outflow[-1], plt_deviations[-1]))

	pyplot.plot(plt_pressure, plt_outflow)
	pyplot.show()
	return

	print(x.shape)
	print(xx.shape)
	print(y.shape)
	print(yy.shape)
	print(p.shape)

	xxt = xx.transpose()
	yyt = yy.transpose()
	ut = u.transpose()
	uut = numpy.zeros((y.size, x.size))
	uut = uut.transpose()

	for idx, width in enumerate(yyt):
		uut[idx, :] = numpy.interp(yyt[-1], width, ut[idx], 0, 0)
		# uut = numpy.append([uut], numpy.interp(yyt[-1], width, ut[idx], 0, 0))

	uu = uut.transpose()

	sum_u = numpy.zeros((1, nx))

	for i in range(nx):
		sum_u[0, i] = numpy.mean(u[1:-1, i] + v[1:-1, i])

	sum_u = numpy.zeros((1, nx))
	for i in range(nx):
		sum_u[0, i] = numpy.mean(u[1:-1, i])

	# pyplot.subplot(5, 1, 1)
	# pyplot.title("velocity field")
	# pyplot.imshow(uu, cmap='jet')
	# # pyplot.colorbar()
	#
	# pyplot.subplot(5, 1, 2)
	# pyplot.title("pressure field")
	# pyplot.imshow(p, cmap='jet')
	# # pyplot.colorbar()
	#
	# pyplot.subplot(5, 1, 3)
	# pyplot.title("velocity at axis")
	# pyplot.plot(x, u[int(ny/2), :])
	#
	# pyplot.subplot(5, 1, 4)
	# pyplot.title("flow cross section at x")
	# pyplot.plot(x, sum_u[0, :])
	#
	# pyplot.subplot(5, 1, 5)
	# pyplot.title("flow cross section at outlet")
	# pyplot.plot(y, v[:, -1])
	#
	# pyplot.show()

if __name__ == '__main__':
	main()
