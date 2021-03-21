import numpy
from matplotlib import pyplot


def main():
	x = numpy.load("x_coord_array.npy")
	y = numpy.load("y_coord_array.npy")
	xx = numpy.load("xx_coord_array.npy")
	yy = numpy.load("yy_coord_array.npy")
	p = numpy.load("pressure_array.npy")
	u = numpy.load("velocity_u_array.npy")
	v = numpy.load("velocity_v_array.npy")
	nx = len(x)
	ny = len(y)

	print(x.size)
	print(xx.size)
	print(y.size)
	print(yy.size)
	print(p.size)

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

	pyplot.subplot(5, 1, 1)
	pyplot.title("velocity field")
	pyplot.imshow(uu, cmap='jet')
	# pyplot.colorbar()

	pyplot.subplot(5, 1, 2)
	pyplot.title("pressure field")
	pyplot.imshow(p, cmap='jet')
	# pyplot.colorbar()

	pyplot.subplot(5, 1, 3)
	pyplot.title("velocity at axis")
	pyplot.plot(x, u[int(ny/2), :])

	pyplot.subplot(5, 1, 4)
	pyplot.title("flow cross section at x")
	pyplot.plot(x, sum_u[0, :])

	pyplot.subplot(5, 1, 5)
	pyplot.title("flow cross section at outlet")
	pyplot.plot(y, v[:, -1])

	pyplot.show()

if __name__ == '__main__':
	main()
