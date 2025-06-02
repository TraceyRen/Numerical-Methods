# Some methods to solve hyperbolic equations
# implicit scheme, upwind scheme, Courant-Isaacson-Ree (CIR) scheme, Lax-Froedroch scheme, Lax-Wendroff scheme, MacCormack scheme
# by Tianzi Ren

from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np

# initial value
def init(x):  # x in (0,1)
    if((x>=0) & (x<1/3)):
        return -sin(3*pi*x)
    elif((x>=1/3) & (x<2/3)):
        return 1
    elif((x>=2/3) & (x<=1)):
        return 0

dt = 0.0005
dx = 0.001

# Upwind scheme, l2 stable when 0 <= dt/dx <= 1
def upwind(dx, dt):
    N = int(10/dt)
    J = int(1/dx)
    M = [[0 for x in range(J)] for y in range(N)]
    r = dt/dx

    for j in range(J):
        M[0][j] = init(j*dx)

    for n in range(N-1):
        for j in range(1,J):
            M[n+1][j] = (1-r)*M[n][j] + r*M[n][j-1]
        M[n+1][0] = M[n+1][J-1]

    return M[-1]

U_upwind = upwind(dx,dt)
plt.plot(np.arange(0, 1, dx), U_upwind)


# CIR, l2 stable when 0 <= dt/dx <= 1
# a = 1, dt < dx, tilde_x between x_{j-1} and x_j
def CIR(dx,dt):
    N = int(10 / dt)
    J = int(1 / dx)
    M = [[0 for x in range(J)] for y in range(N)]

    for j in range(J):
        M[0][j] = init(j * dx)

    for n in range(N - 1):
        for j in range(1, J):
            tilde_x = j * dx - dt
            M[n + 1][j] = (j * dx - tilde_x) / dx * M[n][j - 1] + (tilde_x - (j - 1) * dx) / dx * M[n][j]
        M[n + 1][0] = M[n + 1][J - 1]

    return M[-1]

U_CIR = CIR(dx, dt)
plt.plot(np.arange(0, 1, dx), U_CIR)

# Lax-Friedrichs Scheme, l-infinite stable when |dt/dx| <= 1
def L_F(dx,dt):
    N = int(10 / dt)
    J = int(1 / dx)
    M = [[0 for x in range(J)] for y in range(N)]
    r = dt / dx

    for j in range(J):
        M[0][j] = init(j * dx)

    tilde_x = (J-1) * dx - dt
    for n in range(N - 1):
        for j in range(1, J):
            if(j < J-1):
                M[n + 1][j] = 1 / 2 * (1 - r) * M[n][j + 1] + 1 / 2 * (1 + r) * M[n][j - 1]
            else:
                M[n + 1][j] = (j * dx - tilde_x) / dx * M[n][j - 1] + (tilde_x - (j - 1) * dx) / dx * M[n][j]
        M[n + 1][0] = M[n + 1][J - 1]

    return M[-1]

U_LF = L_F(dx, dt)
plt.plot(np.arange(0, 1, dx), U_LF)

# Lax-Wendroff Scheme, l-2 stable when |dt/dx| <= 1
def L_W(dx, dt):
    N = int(10 / dt)
    J = int(1 / dx)
    M = [[0 for x in range(J)] for y in range(N)]
    r = dt / dx

    for j in range(J):
        M[0][j] = init(j * dx)

    tilde_x = (J-1) * dx - dt
    for n in range(N - 1):
        for j in range(1, J):
            if(j < J-1):
                M[n + 1][j] = M[n][j] - r/2*(M[n][j + 1] - M[n][j-1]) + r**2 / 2 * (M[n][j + 1] - 2*M[n][j] + M[n][j-1])
            else:
                M[n + 1][j] = (j * dx - tilde_x) / dx * M[n][j - 1] + (tilde_x - (j - 1) * dx) / dx * M[n][j]
        M[n + 1][0] = M[n + 1][J - 1]

    return M[-1]

U_LW = L_W(dx, dt)
plt.plot(np.arange(0, 1, dx), U_LW)

# MacCormack Scheme, l-2 stable when |dt/dx| <= 1
def MC(dx, dt):
    N = int(10 / dt)
    J = int(1 / dx)
    M = [[0 for x in range(J)] for y in range(N)]
    r = dt / dx

    for j in range(J):
        M[0][j] = init(j * dx)

    for n in range(N - 1):
        u1_star = [0 for x in range(J)]
        for j in range(1, J):
            u1_star[j] = (1 - r) * M[n][j] + r * M[n][j - 1]
        u1_star[0] = u1_star[J-1]
        for j in range(J-1):
            u2_star = (1 + r) * u1_star[j] - r * u1_star[j+1]
            M[n + 1][j] = 1/2*(M[n][j] + u2_star)
        M[n + 1][J-1] = M[n + 1][0]

    return M[-1]

U_MC = MC(dx, dt)
plt.plot(np.arange(0, 1, dx), U_MC)


# One-side implicit scheme, unconditional stable in l-infinite
def implicit(dx, dt):
    N = int(10 / dt)
    J = int(1 / dx)
    M = np.zeros((J-1,J-1))
    r = dt / dx
    M[0][0] = 1 + r
    M[0][-1] = -r
    for j in range(1, J - 1):
        M[j][j - 1] = -r
        M[j][j] = 1 + r

    M_inv = np.linalg.inv(M)

    U = []
    u0 = np.empty((J-1))
    for j in range(1,J):
        u0[j-1] = init(j * dx)

    U.append(np.insert(u0, 0, init(0)))

    for n in range(1, N):
        u1 = M_inv @ u0
        u0 = u1
        U.append(np.insert(u0, 0, u0[-1]))

    return U[-1]

U_implicit = implicit(dx, dt)
plt.plot(np.arange(0, 1, dx), U_implicit)

# Exact solution
def Exact(x, t):
    a = x-t - np.floor(x-t)
    return init(a)

exact = [Exact(j*dx, 10) for j in range(int(1 / dx))]

# Plot numerical solution along with exact solution at T=10
def overlay(numerical, exact, title):
    plt.plot(np.arange(0, 1, dx), numerical, label = title)
    plt.plot(np.arange(0, 1, dx), exact, label = 'Exact')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()

overlay(U_upwind, exact, 'Upwind scheme')
overlay(U_CIR, exact, 'CIR scheme')
overlay(U_LF, exact, 'Lax-Friedrichs scheme')
overlay(U_LW, exact, 'Lax-Wendroff scheme')
overlay(U_MC, exact, 'MacCormack scheme')
overlay(U_implicit, exact, 'One-side implicit scheme')