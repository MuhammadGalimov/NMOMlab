import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import math

L = 0.33
R = 0.07
n = 10
m = 10
dx = L / (m-1)
dy = 2 * R / (n-1)
mu = 1.0019e-3
ro = 1e3
maxiteration = 500
yx = dy/dx
xy = dx/dy
p2 = 100000
p1 = p2+2.897e-4

p_zv = np.zeros(shape=(n,m)) # начальное значение давления
p = np.zeros(shape=(n,m))
p_zv[:]= p2
p_zv[:,0] = p1
#print(p_zv)

u = np.zeros(shape=(n,m-1))
uOld = np.zeros(shape=(n,m-1))
uAP = np.zeros(shape=(n,m-1))

v = np.zeros(shape=(n+1,m))
vOld = np.zeros(shape=(n+1,m))
vAP = np.zeros(shape=(n+1,m))


#################################################################################
# Функций для u
#################################################################################
def u_aE(u, i, j):
    De = mu * yx
    Fe = ro * dy * (u[i, j] + u[i, j + 1]) / 2
    return De + max(-Fe, 0)


def u_aEleft(u, i, j):
    De = mu * yx
    De2 = mu * yx / 3
    Fe = ro * dy * (u[i, j] + u[i, j + 1]) / 2
    return De + max(-Fe, 0) + De2


def u_aW(u, i, j):
    Dw = mu * yx
    Fw = ro * dy * (u[i, j] + u[i, j - 1]) / 2
    return Dw + max(Fw, 0)


def u_aWright(u, i, j):
    Dw = mu * yx
    Dw2 = mu * yx / 3
    Fw = ro * dy * (u[i, j] + u[i, j - 1]) / 2
    return Dw + max(Fw, 0) + Dw2


def u_aN(v, i, j):
    Dn = mu * xy
    Fn = ro * dx * (v[i, j] + v[i, j + 1]) / 2
    return Dn + max(-Fn, 0)


def u_aS(v, i, j):
    Ds = mu * xy
    Fs = ro * dx * (v[i + 1, j] + v[i + 1, j + 1]) / 2
    return Ds + max(Fs, 0)


#################################################################################
# Функций для v
#################################################################################
def v_aE(u, i, j):
    De = mu * yx
    Fe = ro * dy * (u[i - 1, j] + u[i, j]) / 2
    return De + max(-Fe, 0)


def v_aEleft(u, i, j):
    De = mu * yx
    De2 = mu * yx
    Fe = ro * dy * (u[i - 1, j] + u[i, j]) / 2
    return De + max(-Fe, 0) + De2


def v_aW(u, i, j):
    Dw = mu * yx
    Fw = ro * dy * (u[i - 1, j - 1] + u[i, j - 1]) / 2
    return Dw + max(Fw, 0)


def v_aWright(u, i, j):
    Dw = mu * yx
    Dw2 = mu * yx
    Fw = ro * dy * (u[i - 1, j - 1] + u[i, j - 1]) / 2
    return Dw + max(Fw, 0) + Dw2


def v_aN(v, i, j):
    Dn = mu * xy
    Fn = ro * dx * (v[i, j] + v[i - 1, j]) / 2
    return Dn + max(-Fn, 0)


def v_aS(v, i, j):
    Ds = mu * xy
    Fs = ro * dx * (v[i, j] + v[i + 1, j]) / 2
    return Ds + max(Fs, 0)


def sor(AA, B, omega, initial_guess, convergence_criteria, n):
    iteration = 0
    A = np.copy(AA)
    b = np.copy(B)
    phi = np.copy(initial_guess)
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    while residual > convergence_criteria:
        iteration += 1
        if iteration < n:
            print(iteration)
        else:
            print('exit')
            break
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i, j] * phi[j]
            phi[i] = (1 - omega) * phi[i] + (omega / A[i, i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, phi) - b)
        # print('Residual: {0:10.6g}'.format(residual))
    return phi


iteration = 0
while iteration < maxiteration:

    #############################################################################
    # Построим матрицу и правую часть СЛАУ для нахождения u
    #############################################################################
    U = np.zeros(shape=(n * (m - 1), n * (m - 1)))
    bu = []
    utemp = np.zeros(shape=(n, m - 1))

    for i in range(n):
        for j in range(m - 1):
            if i == 0 or i == n - 1:  # верхний или нижний слой, где горизонтальная скорость задана (u=0)
                utemp[i, j] = 1
                uAP[i, j] = u_aS(vOld, i, j) + u_aN(vOld, i, j) + 2 * mu * yx
                bu.append(0)

                utemp = np.ravel(utemp)
                U[i * (m - 1) + j] = utemp
                utemp = utemp.reshape((n, m - 1))
                utemp.fill(0)
                continue
            if i > 0 and i < n - 1 and j == 0:
                utemp[i, j] = u_aEleft(uOld, i, j) + mu * 3 * yx + u_aN(vOld, i, j) + u_aS(vOld, i, j)
                uAP[i, j] = utemp[i, j]
                utemp[i - 1, j] = -u_aN(vOld, i, j)
                utemp[i, j + 1] = -u_aEleft(uOld, i, j)
                utemp[i + 1, j] = -u_aS(vOld, i, j)
                bu.append((p_zv[i, j] - p_zv[i, j + 1]) * dy)

                utemp = np.ravel(utemp)
                U[i * (m - 1) + j] = utemp
                utemp = utemp.reshape((n, m - 1))
                utemp.fill(0)
                continue
            if i > 0 and i < n - 1 and j == m - 2:
                utemp[i, j] = u_aWright(uOld, i, j) + mu * 3 * yx + u_aN(vOld, i, j) + u_aS(vOld, i, j)
                uAP[i, j] = utemp[i, j]
                utemp[i, j - 1] = -u_aWright(uOld, i, j)
                utemp[i - 1, j] = -u_aN(vOld, i, j)
                utemp[i + 1, j] = -u_aS(vOld, i, j)
                bu.append((p_zv[i, j] - p_zv[i, j + 1]) * dy)

                utemp = np.ravel(utemp)
                U[i * (m - 1) + j] = utemp
                utemp = utemp.reshape((n, m - 1))
                utemp.fill(0)
                continue
            else:
                utemp[i, j] = u_aE(uOld, i, j) + u_aW(uOld, i, j) + u_aN(vOld, i, j) + u_aS(vOld, i, j)
                uAP[i, j] = utemp[i, j]
                utemp[i, j - 1] = -u_aW(uOld, i, j)
                utemp[i, j + 1] = -u_aE(uOld, i, j)
                utemp[i - 1, j] = -u_aN(vOld, i, j)
                utemp[i + 1, j] = -u_aS(vOld, i, j)
                bu.append((p_zv[i, j] - p_zv[i, j + 1]) * dy)

                utemp = np.ravel(utemp)
                U[i * (m - 1) + j] = utemp
                utemp = utemp.reshape((n, m - 1))
                utemp.fill(0)
                continue

    #############################################################################
    # Построим матрицу и правую часть СЛАУ для нахождения v
    #############################################################################
    V = np.zeros(shape=((n + 1) * m, (n + 1) * m))
    bv = []
    vtemp = np.zeros(shape=(n + 1, m))

    for i in range(n + 1):
        for j in range(m):
            if i == 0 or i == n:
                vtemp[i, j] = 1
                vAP[i, j] = 2 * mu * yx + 2 * mu * xy
                bv.append(0)

                vtemp = np.ravel(vtemp)
                V[i * m + j] = vtemp
                vtemp = vtemp.reshape((n + 1, m))
                vtemp.fill(0)
                continue
            if i > 0 and i < n and j == 0:
                vtemp[i, j] = v_aEleft(uOld, i, j) + mu * 4 * yx + v_aN(vOld, i, j) + v_aS(vOld, i, j)
                vAP[i, j] = vtemp[i, j]
                vtemp[i - 1, j] = -v_aN(vOld, i, j)
                vtemp[i, j + 1] = -v_aEleft(uOld, i, j)
                vtemp[i + 1, j] = -v_aS(vOld, i, j)
                bv.append((p_zv[i, j] - p_zv[i - 1, j]) * dy)

                vtemp = np.ravel(vtemp)
                V[i * m + j] = vtemp
                vtemp = vtemp.reshape((n + 1, m))
                vtemp.fill(0)
                continue
            if i > 0 and i < n and j == m - 1:
                vtemp[i, j] = v_aWright(uOld, i, j) + mu * 4 * yx + v_aN(vOld, i, j) + v_aS(vOld, i, j)
                vAP[i, j] = vtemp[i, j]
                vtemp[i, j - 1] = -v_aWright(uOld, i, j)
                vtemp[i - 1, j] = -v_aN(vOld, i, j)
                vtemp[i + 1, j] = -v_aS(vOld, i, j)
                bv.append((p_zv[i, j] - p_zv[i - 1, j]) * dy)

                vtemp = np.ravel(vtemp)
                V[i * m + j] = vtemp
                vtemp = vtemp.reshape((n + 1, m))
                vtemp.fill(0)
                continue
            else:
                vtemp[i, j] = v_aE(uOld, i, j) + v_aW(uOld, i, j) + v_aN(vOld, i, j) + v_aS(vOld, i, j)
                vAP[i, j] = vtemp[i, j]
                vtemp[i, j - 1] = -v_aW(uOld, i, j)
                vtemp[i, j + 1] = -v_aE(uOld, i, j)
                vtemp[i - 1, j] = -v_aN(vOld, i, j)
                vtemp[i + 1, j] = -v_aS(vOld, i, j)
                bv.append((p_zv[i, j] - p_zv[i - 1, j]) * dy)

                vtemp = np.ravel(vtemp)
                V[i * m + j] = vtemp
                vtemp = vtemp.reshape((n + 1, m))
                vtemp.fill(0)
                continue

    #############################################################################
    # Решим две полученные системы уравнений
    #############################################################################
    # u_zv = np.linalg.solve(U,bu)
    # v_zv = np.linalg.solve(V,bv)
    residual_convergence = 1e-8
    omega = 0.8
    initial_guess_u = np.zeros(n * (m - 1))
    initial_guess_v = np.zeros((n + 1) * m)
    u_zv = sor(U, bu, omega, initial_guess_u, residual_convergence, 5)
    v_zv = sor(V, bv, omega, initial_guess_v, residual_convergence, 5)

    u_zv = u_zv.reshape((n, m - 1))
    v_zv = v_zv.reshape((n + 1, m))

    #############################################################################
    # Найдем поправку к давлению
    #############################################################################
    P = np.zeros(shape=(n * m, n * m))
    bp = []
    ptemp = np.zeros(shape=(n, m))
    b = 0
    Bnorm = 0

    for i in range(n):
        for j in range(m):
            if j == 0 and i == 0:
                ptemp[i, j] = ro * dy * dy / uAP[i, j] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((-u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if j == 0 and i > 0 and i < n - 1:
                ptemp[i, j] = ro * dy * dy / uAP[i, j] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((-u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if j == 0 and i == n - 1:
                ptemp[i, j] = ro * dy * dy / uAP[i, j] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                b = ro * ((-u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if j == m - 1 and i == 0:
                ptemp[i, j] = ro * dy * dy / uAP[i, j - 1] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((u_zv[i, j - 1]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if j == m - 1 and i > 0 and i < n - 1:
                ptemp[i, j] = ro * dy * dy / uAP[i, j - 1] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((u_zv[i, j - 1]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if j == m - 1 and i == n - 1:
                ptemp[i, j] = ro * dy * dy / uAP[i, j - 1] + ro * dx * dx / vAP[i, j] + ro * dx * dx / vAP[i + 1, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                b = ro * ((u_zv[i, j - 1]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if i == 0 and j > 0 and j < m - 1:
                ptemp[i, j] = ro * (dy * dy / uAP[i, j] + dy * dy / uAP[i, j - 1] + dx * dx / vAP[i, j] + dx * dx / vAP[
                    i + 1, j])
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((u_zv[i, j - 1] - u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            if i == n - 1 and j > 0 and j < m - 1:
                ptemp[i, j] = ro * (dy * dy / uAP[i, j] + dy * dy / uAP[i, j - 1] + dx * dx / vAP[i, j] + dx * dx / vAP[
                    i + 1, j])
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                b = ro * ((u_zv[i, j - 1] - u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue
            else:
                ptemp[i, j] = ro * (dy * dy / uAP[i, j] + dy * dy / uAP[i, j - 1] + dx * dx / vAP[i, j] + dx * dx / vAP[
                    i + 1, j])
                ptemp[i, j + 1] = ro * dy * dy / uAP[i, j]
                ptemp[i, j - 1] = ro * dy * dy / uAP[i, j - 1]
                ptemp[i - 1, j] = ro * dx * dx / vAP[i, j]
                ptemp[i + 1, j] = ro * dx * dx / vAP[i + 1, j]
                b = ro * ((u_zv[i, j - 1] - u_zv[i, j]) * dy + (v_zv[i + 1, j] - v_zv[i, j]) * dx)
                bp.append(b)
                Bnorm += b * b

                ptemp = np.ravel(ptemp)
                P[i * m + j] = ptemp
                ptemp = ptemp.reshape((n, m))
                ptemp.fill(0)
                continue

    initial_guess_p = np.zeros(n * m)
    # p_p = np.linalg.solve(P,bp)
    # u_zv = sor(U, bu, omega, initial_guess_u, residual_convergence, 5)
    p_p = sor(P, bp, omega, initial_guess_p, residual_convergence, 5)

    p_p = p_p.reshape((n, m))
    Bnorm = math.sqrt(Bnorm)

    p = p_zv + 0.7 * p_p

    for i in range(n):
        for j in range(m - 1):
            if i == 0 or i == n - 1:
                continue
            else:
                u[i, j] = u_zv[i, j] + dy * (p_p[i, j] - p_p[i, j + 1]) / uAP[i, j]
                continue

    for i in range(n + 1):
        for j in range(m):
            if i == 0 or i == n:
                continue
            else:
                v[i, j] = v_zv[i, j] + dx * (p_p[i, j] - p_p[i - 1, j]) / vAP[i, j]

    print(Bnorm)

    if Bnorm < 0.00001:
        break
    else:
        p_zv = p
        uOld = u
        vOld = v

    iteration += 1