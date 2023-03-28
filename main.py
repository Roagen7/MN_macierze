import time

import matplotlib
import math
import matplotlib.pyplot as plt
from joblib import dump, load

from copy import deepcopy

matplotlib.use("MacOSX")

f = 8
e = 6
c = 9
d = 7


def print_debug(mat: list):
    for x in mat:
        print(x)


def mat_identity(n):
    return [[0 if x != y else 1 for x in range(n)] for y in range(n)]


def matmul(mat1: list, mat2: list):
    y1 = len(mat1)
    x1 = len(mat1[0])
    x2 = len(mat2[0])
    mat = [[0 for _ in range(x2)] for _ in range(y1)]

    for y in range(len(mat)):
        for x in range(len(mat[0])):
            mat[y][x] = sum([mat1[y][l] * mat2[l][x] for l in range(x1)])

    return mat


def matsub(mat1: list, mat2: list) -> list:
    mat = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
    for y in range(len(mat1)):
        for x in range(len(mat1[0])):
            mat[y][x] = mat1[y][x] - mat2[y][x]
    return mat


def residuum(mat_a: list, b: list, x: list) -> list:
    return matsub(matmul(mat_a, x), b)


def norm(vec: list):
    return sum([y[0]**2 for y in vec])**0.5


def construct(n=9 * 100 + c * 10 + d) -> (list, list):
    a1 = 5 + e
    a2 = -1
    a3 = -1

    mat_a = [[0 for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            if x == y:
                mat_a[y][x] = a1
            elif x == y - 1 or x == y + 1:
                mat_a[y][x] = a2
            elif x == y - 2 or x == y + 2:
                mat_a[y][x] = a3

    b = [[math.sin((i+1) * (f+1))] for i in range(n)]
    return mat_a, b


def lu_decomposition(mat: list) -> (list, list):
    m = len(mat)
    U = deepcopy(mat)
    L = mat_identity(m)

    for k in range(0, m-1):
        for j in range(k+1, m):
            L[j][k] = U[j][k]/U[k][k]
            for i in range(k, m):
                U[j][i] -= L[j][k] * U[k][i]
    return L, U


def solve_upper_triangle(mat: list, b: list) -> list:
    n = len(mat)
    vec_x = [[0] for _ in range(n)]

    vec_x[-1][0] = b[n-1][0]/mat[n-1][n-1]
    for i in range(n-2, -1, -1):
        vec_x[i][0] = (b[i][0] - sum([mat[i][j] * vec_x[j][0] for j in range(n-1, i, -1)]))/mat[i][i]

    return vec_x


def solve_lower_triangle(mat: list, b: list) -> list:
    n = len(mat)
    vec_x = [[0] for _ in range(n)]

    vec_x[0][0] = b[0][0]/mat[0][0]
    for i in range(1, n):
        vec_x[i][0] = (b[i][0] - sum([mat[i][j] * vec_x[j][0] for j in range(0, i)]))/mat[i][i]

    return vec_x


def lu(mat_a: list, b: list, verbose=False) -> list:
    t0 = time.time()
    L, U = lu_decomposition(mat_a)
    t = time.time() - t0
    if verbose:
        print(f"lu decomposition={t}")
    y = solve_lower_triangle(L, b)
    return solve_upper_triangle(U, y)


def jacobi(mat_a: list, b: list,  diverges=False, epsilon=10**-9) -> (list, int):
    iters = 0
    n = len(mat_a)
    vec_x = [[0] for _ in range(n)]
    norms = []

    while norm(residuum(mat_a, b, vec_x)) > epsilon:
        if diverges and iters >= 50:
            break
        new_vec_x = [[0] for _ in range(n)]
        norms.append(norm(residuum(mat_a, b, vec_x)))
        iters += 1
        for i in range(0, n):
            sig = sum([0 if i == j else vec_x[j][0]*mat_a[i][j] for j in range(n)])
            new_vec_x[i][0] = (b[i][0] - sig)/mat_a[i][i]
        for i in range(len(vec_x)):
            vec_x[i][0] = new_vec_x[i][0]

    norms.append(norm(residuum(mat_a, b, vec_x)))
    return vec_x, iters, norms


def gauss_seidel(mat_a: list, b:list, diverges=False, epsilon=10**-9) -> (list, int):
    iters = 0
    n = len(mat_a)
    vec_x = [[0] for _ in range(n)]
    norms = []

    while norm(residuum(mat_a, b, vec_x)) > epsilon:
        if diverges and iters >= 50:
            break
        new_vec_x = [[0] for _ in range(n)]
        norms.append(norm(residuum(mat_a, b, vec_x)))
        iters += 1
        for i in range(0, n):
            sig1 = sum([mat_a[i][j] * new_vec_x[j][0] for j in range(0, i)])
            sig2 = sum([mat_a[i][j] * vec_x[j][0] for j in range(i+1, n)])
            new_vec_x[i][0] = (b[i][0] - sig1 - sig2)/mat_a[i][i]
        for i in range(len(vec_x)):
            vec_x[i][0] = new_vec_x[i][0]

    norms.append(norm(residuum(mat_a, b, vec_x)))
    return vec_x, iters, norms


def zb():
    import time
    m_a, vec_b = construct()

    j_t0 = time.time()
    j_x, j_iters, j_norms = jacobi(m_a, vec_b)
    j_t = time.time() - j_t0

    gs_t0 = time.time()
    gs_x, gs_iters, gs_norms = gauss_seidel(m_a, vec_b)
    gs_t = time.time() - gs_t0

    lu_t0 = time.time()
    gauss_x = lu(m_a, vec_b)
    gauss_t = time.time() - lu_t0

    print(f"Jacobi: iters={j_iters} residuum={norm(residuum(m_a, vec_b, j_x))} t={j_t}")
    print(f"Gauss-Seidel: iters={gs_iters} residuum={norm(residuum(m_a, vec_b, gs_x))} t={gs_t}")
    print(f"LU: residuum={norm(residuum(m_a, vec_b, gauss_x))} t={gauss_t}")

    fig, axis = plt.subplots(2)
    plt.suptitle("błąd rezydualny dla metod iteracyjnych w i-tej iteracji")
    axis[0].plot([i for i in range(len(j_norms))], j_norms)
    axis[0].set_yscale("log")
    axis[0].set_title("Metoda Jacobiego")
    axis[0].set_ylabel("norm(res)")
    axis[0].set_xlabel("ilość iteracji")
    axis[1].plot([i for i in range(len(gs_norms))], gs_norms)
    axis[1].set_yscale("log")
    axis[1].set_title("Metoda Gaussa-Seidla")
    axis[1].set_ylabel("norm(res)")
    axis[1].set_xlabel("ilość iteracji")
    plt.show()


def argmin_min(arr):
    m = arr[0]
    m_i = 0
    for i in range(len(arr)):
        if arr[i] < m:
            m = arr[i]
            m_i = i
    return m_i, m

def zc():
    m_a, vec_b = construct()
    for i in range(len(m_a)):
        m_a[i][i] = 3

    fig, axis = plt.subplots(2)

    j_x, j_iters, j_norms = jacobi(m_a, vec_b, diverges=True)
    gs_x, gs_iters, gs_norms = gauss_seidel(m_a, vec_b, diverges=True)
    j_mini, j_min = argmin_min(j_norms)
    gs_mini, gs_min = argmin_min(gs_norms)

    plt.suptitle("błąd rezydualny dla metod iteracyjnych")
    axis[0].plot([i for i in range(len(j_norms))], j_norms)
    axis[0].scatter(j_mini, j_min, color='red')
    axis[0].set_xticks([i for i in range(0, 60, 10)] + [j_mini])
    axis[0].set_yscale("log")
    axis[0].set_title("Metoda Jacobiego")
    axis[0].set_ylabel("norm(res)")
    axis[0].set_xlabel("ilość iteracji")
    axis[1].plot([i for i in range(len(gs_norms))], gs_norms)
    axis[1].scatter(gs_mini, gs_min, color='red')
    axis[1].set_xticks([i for i in range(0, 60, 10)] + [gs_mini])
    axis[1].set_yscale("log")
    axis[1].set_title("Metoda Gaussa-Seidla")
    axis[1].set_ylabel("norm(res)")
    axis[1].set_xlabel("ilość iteracji")
    plt.show()


def zd():
    import time
    m_a, vec_b = construct()
    for i in range(len(m_a)):
        m_a[i][i] = 3
    lu_t0 = time.time()
    gauss_x = lu(m_a, vec_b)
    gauss_t = time.time() - lu_t0
    print(f"LU: residuum={norm(residuum(m_a, vec_b, gauss_x))} t={gauss_t}")


def ze_count():
    j_times = []
    j_iters = []
    gs_times = []
    gs_iters = []
    lu_times = []

    for N in [100, 500, 1000, 1500, 2000]:
        m_a, vec_b = construct(N)

        j_t0 = time.time()
        _, j_iter, _ = jacobi(m_a, vec_b)
        j_times.append(time.time() - j_t0)
        j_iters.append(j_iter)

        gs_t0 = time.time()
        _, gs_iter, _ = gauss_seidel(m_a, vec_b)
        gs_times.append(time.time() - gs_t0)
        gs_iters.append(gs_iter)

        lu_t0 = time.time()
        lu(m_a, vec_b)
        lu_times.append(time.time() - lu_t0)

        print(f"{N} finished")
        dump((j_times, gs_times, lu_times), 'times.joblib')
        dump((j_iters, gs_iters), 'iters.joblib')

    print((j_times, gs_times, lu_times))
    print((j_iters, gs_iters))


def ze_plot():
    N = [100, 500, 1000, 1500, 2000]
    j_times, gs_times, lu_times = load('times.joblib')
    j_iters, gs_iters = load('iters.joblib')

    fig, axis = plt.subplots(3)
    plt.suptitle("czasy wykonania algorytmów dla różnych rozmiarów macierzy")
    axis[0].plot(N, j_times)
    axis[0].set_title("Metoda Jacobiego")
    axis[0].set_ylabel("t [s]")
    axis[0].set_xlabel("N")
    axis[1].plot(N, gs_times)
    axis[1].set_title("Metoda Gaussa-Seidla")
    axis[1].set_ylabel("t [s]")
    axis[1].set_xlabel("N")
    axis[2].plot(N, lu_times)
    axis[2].set_title("Metoda faktoryzacji LU")
    axis[2].set_ylabel("t [s]")
    axis[2].set_xlabel("N")
    plt.show()

    fig, axis = plt.subplots(2)
    plt.suptitle("ilość iteracji metody Jacobiego i Gaussa-Seidla dla różnych rozmiarów macierzy")
    axis[0].scatter(N, j_iters)
    axis[0].set_title("Metoda Jacobiego")
    axis[0].set_ylabel("ilość iteracji")
    # axis[0].set_yticks([j_iters[0]])
    axis[0].set_xlabel("N")
    axis[1].scatter(N, gs_iters)
    axis[1].set_title("Metoda Gaussa-Seidla")
    axis[1].set_ylabel("ilość iteracji")
    # axis[1].set_yticks([gs_iters[0]])
    axis[1].set_xlabel("N")
    plt.show()


def ze_extra():
    j_iters = []
    gs_iters = []
    m_a, vec_b = construct(100)
    E = [10**-1, 10**-4, 10**-8, 10**-9, 10**-11]

    for e in E:
        _, j_iter, _ = jacobi(m_a, vec_b, epsilon=e)
        _, gs_iter, _ = gauss_seidel(m_a, vec_b, epsilon=e)
        j_iters.append(j_iter)
        gs_iters.append(gs_iter)

    fig, axis = plt.subplots(2)
    plt.suptitle("ilość iteracji metody Jacobiego i Gaussa-Seidla dla różnych dokładności (N=100)")
    axis[0].plot(E, j_iters)
    axis[0].set_title("Metoda Jacobiego")
    axis[0].set_ylabel("ilość iteracji")
    axis[0].set_xlabel("oczekiwane norm(res)")
    axis[0].set_xscale("log")
    axis[0].invert_xaxis()
    axis[1].plot(E, gs_iters)
    axis[1].set_title("Metoda Gaussa-Seidla")
    axis[1].set_ylabel("ilość iteracji")
    axis[1].set_xlabel("oczekiwane norm(res)")
    axis[1].set_xscale("log")
    axis[1].invert_xaxis()
    plt.show()


if __name__ == '__main__':
    ze_plot()
