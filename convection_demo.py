import numpy as np


def navier_stokes_initialisation(niter, r, nx_or_ny, tmax, xmax_or_ymax, nu):
    """
    Initialise conditions for Poisson Equation
    """
    # Increments:
    nx = ny = nx_or_ny
    xmax = ymax = xmax_or_ymax
    dx = xmax / (nx - 1)
    dy = ymax / (ny - 1)
    nt = int(((nu * tmax) / (r * (dx) ** 2)) + 1)
    dt = tmax / (nt - 1)

    # Initialise data structures:
    import numpy as np

    # Initial conditions
    p = np.zeros((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))

    # x and y range
    x = np.zeros(nx)
    y = np.zeros(ny)

    # X Loop
    x = np.linspace(0.0, 2.0, nx)

    # Y Loop
    y = np.linspace(0.0, 2.0, ny)

    return p, x, y, u, v, nx, ny, nt, dx, dy, dt, niter, r


def navier_stokes_equation_accurate(niter, nx, rho, nu, r, tmax):
    (p, x, y, u, v, nx, ny, nt,
     dx, dy, dt, niter, r) = navier_stokes_initialisation(niter, r, nx, tmax, 2.0, nu)

    # Increments
    h = dx

    import numpy as np

    # Intermediate copies:
    un = np.zeros((nx, ny))
    vn = np.zeros((nx, ny))
    pm = np.zeros((nx, ny))
    bm = np.zeros((nx, ny))  # bm needs to be exactly zero at the boundaries

    # Loop - use decimal points for all floating point numbers
    # for n in range(nt):

    udiff = 1.0

    while udiff > 0.00001:

        # First points for bm. We don't know the value at i=0 and i=nx-1. b is zero at j=0 and j=ny-1.
        i = 1
        j = 1
        bm[i:nx - 1, j:ny - 1] = ((rho / (2.0 * h * dt)) * (u[i + 1:nx, j:ny - 1] - u[i - 1:nx - 2, j:ny - 1]
                                                            + v[i:nx - 1, j + 1:ny] - v[i:nx - 1, j - 1:ny - 2]) +
                                  (rho / (4.0 * h ** 2)) * ((u[i + 1:nx, j:ny - 1] - u[i - 1:nx - 2, j:ny - 1]) ** 2.0 +
                                                            4.0 * h * (u[i:nx - 1, j + 1:ny] - u[i:nx - 1,
                                                                                               j - 1:ny - 2]) *
                                                            (v[i + 1:nx, j:ny - 1] - v[i - 1:nx - 2, j:ny - 1]) +
                                                            (v[i:nx - 1, j + 1:ny] - v[i:nx - 1, j - 1:ny - 2]) ** 2.0))
        i = 0
        # Periodic at i = 0 Replace all the i=i-1 with i=nx-1
        bm[i, j:ny - 1] = ((rho / (2.0 * h * dt)) * (u[i + 1, j:ny - 1] - u[nx - 1, j:ny - 1]
                                                     + v[i, j + 1:ny] - v[i, j - 1:ny - 2]) +
                           (rho / (4.0 * h ** 2)) * ((u[i + 1, j:ny - 1] - u[nx - 1, j:ny - 1]) ** 2.0 +
                                                     4.0 * h * (u[i, j + 1:ny] - u[i, j - 1:ny - 2]) *
                                                     (v[i + 1, j:ny - 1] - v[nx - 1, j:ny - 1]) +
                                                     (v[i, j + 1:ny] - v[i, j - 1:ny - 2]) ** 2.0))

        i = nx - 1
        # Periodic at i = nx-1 Replace all the i=i+1 with i=0
        bm[i, j:ny - 1] = ((rho / (2.0 * h * dt)) * (u[0, j:ny - 1] - u[i - 1, j:ny - 1]
                                                     + v[i, j + 1:ny] - v[i, j - 1:ny - 2]) +
                           (rho / (4.0 * h ** 2)) * ((u[0, j:ny - 1] - u[i - 1, j:ny - 1]) ** 2.0 +
                                                     4.0 * h * (u[i, j + 1:ny] - u[i, j - 1:ny - 2]) *
                                                     (v[0, j:ny - 1] - v[i - 1, j:ny - 1]) +
                                                     (v[i, j + 1:ny] - v[i, j - 1:ny - 2]) ** 2.0))

        for m in range(niter):
            # First points for p. We don't know the pressure at i=0 and i=nx-1. dp/dy = 0 at y=0 and y=2
            pm = np.copy(p)
            i = 1
            j = 1
            p[i:nx - 1, j:ny - 1] = 0.25 * (pm[i + 1:nx, j:ny - 1] + pm[i - 1:nx - 2, j:ny - 1]
                                            + pm[i:nx - 1, j + 1:ny] + pm[i:nx - 1, j - 1:ny - 2]
                                            - bm[i:nx - 1, j:ny - 1] * h ** 2.0)
            i = 0
            # Periodic at i = 0 Replace all the i=i-1 with i=nx-1
            p[i, j:ny - 1] = 0.25 * (pm[i + 1, j:ny - 1] + pm[nx - 1, j:ny - 1]
                                     + pm[i, j + 1:ny] + pm[i, j - 1:ny - 2]
                                     - bm[i, j:ny - 1] * h ** 2.0)
            i = nx - 1
            # Periodic at i = nx-1 Replace all the i=i+1 with i=0
            p[i, j:ny - 1] = 0.25 * (pm[0, j:ny - 1] + pm[i - 1, j:ny - 1]
                                     + pm[i, j + 1:ny] + pm[i, j - 1:ny - 2]
                                     - bm[i, j:ny - 1] * h ** 2.0)

            # Set zero gradient boundary conditions
            p[:, 0] = p[:, 1]
            p[:, ny - 1] = p[:, ny - 2]

        # First points for u and v. We don't know u and v at i=0 and i=nx-1. u and v are zero at j=0 and j=ny-1.
        i = 1
        j = 1

        un = np.copy(u)
        vn = np.copy(v)

        u[i:nx - 1, j:ny - 1] = (un[i:nx - 1, j:ny - 1] -
                                 (dt / h) * (un[i:nx - 1, j:ny - 1] * (
                            un[i:nx - 1, j:ny - 1] - un[i - 1:nx - 2, j:ny - 1]) +
                                             vn[i:nx - 1, j:ny - 1] * (
                                                         un[i:nx - 1, j:ny - 1] - un[i:nx - 1, j - 1:ny - 2])) +
                                 dt - (dt / (2.0 * rho * h)) * (p[i + 1:nx, j:ny - 1] - p[i - 1:nx - 2, j:ny - 1]) +
                                 ((dt * nu) / (h ** 2.0)) * (un[i - 1:nx - 2, j:ny - 1] + un[i + 1:nx, j:ny - 1] +
                                                             un[i:nx - 1, j - 1:ny - 2] + un[i:nx - 1, j + 1:ny] -
                                                             4.0 * un[i:nx - 1, j:ny - 1]))

        v[i:nx - 1, j:ny - 1] = (vn[i:nx - 1, j:ny - 1] -
                                 (dt / h) * (un[i:nx - 1, j:ny - 1] * (
                            vn[i:nx - 1, j:ny - 1] - vn[i - 1:nx - 2, j:ny - 1]) +
                                             vn[i:nx - 1, j:ny - 1] * (
                                                         vn[i:nx - 1, j:ny - 1] - vn[i:nx - 1, j - 1:ny - 2])) -
                                 (dt / (2.0 * rho * h)) * (p[i:nx - 1, j + 1:ny] - p[i:nx - 1, j - 1:ny - 2]) +
                                 ((dt * nu) / (h ** 2.0)) * (vn[i - 1:nx - 2, j:ny - 1] + vn[i + 1:nx, j:ny - 1] +
                                                             vn[i:nx - 1, j - 1:ny - 2] + vn[i:nx - 1, j + 1:ny] -
                                                             4.0 * vn[i:nx - 1, j:ny - 1]))

        # Periodic at i = 0 Replace all the i=i-1 with i=nx-1
        i = 0

        u[i, j:ny - 1] = (un[i, j:ny - 1] -
                          (dt / h) * (un[i, j:ny - 1] * (un[i, j:ny - 1] - un[nx - 1, j:ny - 1]) +
                                      vn[i, j:ny - 1] * (un[i, j:ny - 1] - un[i, j - 1:ny - 2])) +
                          dt - (dt / (2.0 * rho * h)) * (p[i + 1, j:ny - 1] - p[nx - 1, j:ny - 1]) +
                          ((dt * nu) / (h ** 2.0)) * (un[nx - 1, j:ny - 1] + un[i + 1, j:ny - 1] +
                                                      un[i, j - 1:ny - 2] + un[i, j + 1:ny] -
                                                      4.0 * un[i, j:ny - 1]))

        v[i, j:ny - 1] = (vn[i, j:ny - 1] -
                          (dt / h) * (un[i, j:ny - 1] * (vn[i, j:ny - 1] - vn[nx - 1, j:ny - 1]) +
                                      vn[i, j:ny - 1] * (vn[i, j:ny - 1] - vn[i, j - 1:ny - 2])) -
                          (dt / (2.0 * rho * h)) * (p[i, j + 1:ny] - p[i, j - 1:ny - 2]) +
                          ((dt * nu) / (h ** 2.0)) * (vn[i - 1, j:ny - 1] + vn[i + 1, j:ny - 1] +
                                                      vn[i, j - 1:ny - 2] + vn[i, j + 1:ny] -
                                                      4.0 * vn[i, j:ny - 1]))

        # Periodic at i = nx-1 Replace all the i=i+1 with i=0
        i = nx - 1

        u[i, j:ny - 1] = (un[i, j:ny - 1] -
                          (dt / h) * (un[i, j:ny - 1] * (un[i, j:ny - 1] - un[i - 1, j:ny - 1]) +
                                      vn[i, j:ny - 1] * (un[i, j:ny - 1] - un[i, j - 1:ny - 2])) +
                          dt - (dt / (2.0 * rho * h)) * (p[0, j:ny - 1] - p[i - 1, j:ny - 1]) +
                          ((dt * nu) / (h ** 2.0)) * (un[i - 1, j:ny - 1] + un[0, j:ny - 1] +
                                                      un[i, j - 1:ny - 2] + un[i, j + 1:ny] -
                                                      4.0 * un[i, j:ny - 1]))

        v[i, j:ny - 1] = (vn[i, j:ny - 1] -
                          (dt / h) * (un[i, j:ny - 1] * (vn[i, j:ny - 1] - vn[i - 1, j:ny - 1]) +
                                      vn[i, j:ny - 1] * (vn[i, j:ny - 1] - vn[i, j - 1:ny - 2])) -
                          (dt / (2.0 * rho * h)) * (p[i, j + 1:ny] - p[i, j - 1:ny - 2]) +
                          ((dt * nu) / (h ** 2.0)) * (vn[i - 1, j:ny - 1] + vn[0, j:ny - 1] +
                                                      vn[i, j - 1:ny - 2] + vn[i, j + 1:ny] -
                                                      4.0 * vn[i, j:ny - 1]))

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)

    return u, v, p, x, y


def plot_3D(u,x,y,title,label):
    """
    Plots the 2D velocity field
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure(figsize=(11,7),dpi=100)
    ax=fig.gca(projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel(label)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.view_init(30,225)
    Y,X=np.meshgrid(y,x) #note meshgrid uses y,x not x,y!!!
    surf=ax.plot_surface(X,Y,u[:,:], rstride=1, cstride=1)
    plt.title(title)
    plt.show()


def parallel_plates_analytical(YMAX, NY, NU):
    # Initialise data structures
    u_analytical = np.zeros(NY)
    y = np.zeros(NY)

    # Constants
    DY = YMAX / (NY - 1)

    # Y Loop
    for j in range(0, NY):
        y[j] = j * DY

    # Analytical solution
    for j in range(0, NY):
        u_analytical[j] = (1.0 / NU) * (y[j] - (y[j] ** 2.0 / 2.0))

    return u_analytical, y


# def plot_diffusion_2(u1,u2,u3,u4,u5,y1,y2,y3,y4,y5,NX):
# def plot_diffusion_2(u1, u2, y1, y2, NX):
def plot_diffusion_2(u1, u2, u3, y1, y2, y3, NX):
   """
   Plots the 1D velocity field
   """

   import matplotlib.pyplot as plt
   import matplotlib.cm as cm
   plt.figure()
   ax=plt.subplot(111)
   ax.plot(y1,u1[-1,:],'-', markerfacecolor='none', alpha=0.5, label='Numerical nx=51')
   ax.plot(y2,u2[-1,:],'-', markerfacecolor='none', alpha=0.5, label='Numerical nx=41')
   ax.plot(y3,u3[:],linestyle='-',c='r',label='Analytical')
   # ax.plot(y4,u4[-1,:],'-', markerfacecolor='none', alpha=0.5, label='Numerical nx=21')
   # ax.plot(y5,u5[-1,:],'-', markerfacecolor='none', alpha=0.5, label='Numerical nx=11')
   # box=ax.get_position()
   # ax.set_position([box.x0, box.y0, box.width*1.5,box.height*1.5])
   # ax.legend( bbox_to_anchor=(1.02,1), loc=2)
   plt.xlabel('y (m)')
   plt.ylabel('u (m/s)')
   plt.show()


u002, v002, p002, x002, y002 = navier_stokes_equation_accurate(50, 51, 1.0, 0.1, 0.25, 2.0)
u003, v003, p003, x003, y003 = navier_stokes_equation_accurate(50, 41, 1.0, 0.1, 0.25, 2.0)
# plot_diffusion_2(u002,u003,y002,y003,51)

u_analytic, y_analytic = parallel_plates_analytical(2.0, 51, 0.1)
plot_diffusion_2(u002,u003,u_analytic,y002,y003,y_analytic,51)

# u004, v004, p004, x004, y004 = navier_stokes_equation_accurate(50, 31, 1.0, 0.1, 0.5, 2.0)
# u005, v005, p005, x005, y005 = navier_stokes_equation_accurate(50, 21, 1.0, 0.1, 0.5, 2.0)
# u006, v006, p006, x006, y006 = navier_stokes_equation_accurate(50, 11, 1.0, 0.1, 0.5, 2.0)

# plot_diffusion_2(u002,u003,u004,u005,u006,y002,y003,y004,y005,y006,51)


# u60, v60, p60, x60, y60 = navier_stokes_equation_accurate(50, 51, 1.0, 0.1, 0.5, 2.0)
# plot_3D(u60,x60,y60,'Figure 1: Final Conditions','u (m/s)')
# plot_3D(v60,x60,y60,'Figure 2: Final Conditions','v (m/s)')
# plot_3D(p60,x60,y60,'Figure 3: Final Conditions','p (Pa)')


