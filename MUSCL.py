# MUSCL
# by Tianzi Ren

from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np

# minimod function
def minimod(a, b):
    if((a>=0) & (b>=0)):
        return min(a, b)
    elif((a<=0) & (b<=0)):
        return max(a, b)
    else:
        return 0

# (a)
# initial value
def init(x):  # x in (0,1)
    if((x>=0) & (x<1/3)):
        return -sin(3*pi*x)
    elif((x>=1/3) & (x<2/3)):
        return 1
    elif((x>=2/3) & (x<=1)):
        return 0

# flux function
def godunov_u(u, v):
    return u

# MUSCL
# compute next time level U value
def next_time_level(current):
    # INPUT: current is a vector, size (J+1)
    # tilde U
    tilde_U = []
    for j in range(J+1):
        if(j == 0):
            slope = (current[1] - current[0]) / dx
        elif(j == J):
            slope = ((current[J] - current[J - 1]) / dx)
        else:
            slope = minimod((current[j + 1] - current[j]) / dx, (current[j] - current[j - 1]) / dx)
        j_l = current[j] + slope * (- 1 / 2) * dx
        j_r = current[j] + slope * 1 / 2 * dx
        tilde_U += [j_l, j_r]

    U_1 = [None] * (J+1)
    for j in range(1, J):
        # flux
        F_l = godunov_u(tilde_U[j*2-1], tilde_U[j*2])
        F_r = godunov_u(tilde_U[j*2+1], tilde_U[(j+1)*2])

        U_1[j] = current[j] - dt/dx * (F_r - F_l)

    # boundary
    tilde_xJ = J * dx - dt
    U_1[J] = (J * dx - tilde_xJ) / dx * current[J - 1] + (tilde_xJ - (J - 1) * dx) / dx * current[J]

    U_1[0] = U_1[J]

    return U_1

# initial
dt = 0.0005
dx = 0.001
r = dt/dx
N = int(10/dt)
J = int(1/dx)
M = [[0 for x in range(J+1)] for y in range(N+1)]

for j in range(J+1):
    M[0][j] = init(j*dx)

for n in range(N):
    U_1 = next_time_level(M[n])
    U_2 = next_time_level(U_1)

    M[n+1] = [0.5*(u1 + u2) for u1, u2 in zip(M[n], U_2)]

plt.plot(np.linspace(0, 1, J+1), M[-1])

# Exact solution
def Exact(x, t):
    a = x-t - np.floor(x-t)
    return init(a)

exact = [Exact(j*dx, 10) for j in range(int(1 / dx)+1)]

# Plot numerical solution along with exact solution at T=10
def overlay(numerical, exact, title):
    plt.plot(np.linspace(0, 1, J+1), numerical, label = title)
    plt.plot(np.linspace(0, 1, J+1), exact, label = 'Exact')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()

overlay(M[-1], exact, 'MUSCL')


# (b)
# initial value
def init(x):
    return 1+sin(2*pi*x)

# flux function
def godunov_u2(u, v):
    if ((u >= 0) & (v >= 0)):
        return 1/2*u**2
    elif ((u <= 0) & (v <= 0)):
        return 1/2*v**2
    elif ((u >= 0) & (v < 0)):
        return max(1/2*u**2, 1/2*v**2)
    elif ((u <= 0) & (v > 0)):
        return 0

dt = 0.0005
dx = 0.001
r = dt/dx
N = int(0.5/dt)
J = int(10/dx)
M = [[0 for x in range(J+1)] for y in range(N+1)]

for j in range(J+1):
    M[0][j] = init(j*dx)

# compute next time level U value
def next_time_level(current):
    # INPUT: current is a vector, size (J+1)
    # tilde U
    tilde_U = []
    for j in range(J+1):
        if(j == 0):
            slope = (current[1] - current[0]) / dx
        elif(j == J):
            slope = ((current[J] - current[J - 1]) / dx)
        else:
            slope = minimod((current[j + 1] - current[j]) / dx, (current[j] - current[j - 1]) / dx)
        j_l = current[j] + slope * (- 1 / 2) * dx
        j_r = current[j] + slope * 1 / 2 * dx
        tilde_U += [j_l, j_r]

    U_1 = [None] * (J+1)
    for j in range(1, J):
        # flux
        F_l = godunov_u2(tilde_U[j*2-1], tilde_U[j*2])
        F_r = godunov_u2(tilde_U[j*2+1], tilde_U[(j+1)*2])

        U_1[j] = current[j] - dt/dx * (F_r - F_l)

    # boundary
    a = current[J - 1]
    tilde_xJ = J * dx - a * dt
    U_1[J] = (J * dx - tilde_xJ) / dx * current[J - 1] + (tilde_xJ - (J - 1) * dx) / dx * current[J]

    U_1[0] = U_1[J]

    return U_1

u1 = next_time_level(M[0])
plt.plot(np.linspace(0, 1, J+1), u1)

for n in range(N):
    U_1 = next_time_level(M[n])
    U_2 = next_time_level(U_1)

    M[n+1] = [0.5*(u1 + u2) for u1, u2 in zip(M[n], U_2)]

plt.plot(np.linspace(1, 2, int(1/dx)+1), M[-1][1000: 2001], label = "MUSCL solution")
plt.plot(np.linspace(1, 2, int(1/dx)+1), M[0][1000: 2001], label = "Initial value")
plt.title("One period u(x)")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()