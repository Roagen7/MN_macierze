import matplotlib
import math
import matplotlib.pyplot as plt

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


def construct() -> (list, list):
    n = 9 * 100 + c * 10 + d # N = 9cd
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

    b = [[math.sin((i+1) * f)] for i in range(n)]
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


def lu(mat_a: list, b: list) -> list:
    L, U = lu_decomposition(mat_a)
    y = solve_lower_triangle(L, b)
    return solve_upper_triangle(U, y)


def jacobi(mat_a: list, b: list, epsilon=10**-9) -> (list, int):
    iters = 0
    n = len(mat_a)
    vec_x = [[0] for _ in range(n)]
    norms = []

    while norm(residuum(mat_a, b, vec_x)) > epsilon and iters < 50:
        new_vec_x = [[0] for _ in range(n)]
        norms.append(norm(residuum(mat_a, b, vec_x)))
        iters += 1
        for i in range(0, n):
            sig = sum([0 if i == j else vec_x[j][0]*mat_a[i][j] for j in range(n)])
            new_vec_x[i][0] = (b[i][0] - sig)/mat_a[i][i]
        for i in range(len(vec_x)):
            vec_x[i][0] = new_vec_x[i][0]

    return vec_x, iters, norms


def gauss_seidel(mat_a: list, b:list, epsilon=10**-9) -> (list, int):
    iters = 0
    n = len(mat_a)
    vec_x = [[0] for _ in range(n)]
    norms = []

    while norm(residuum(mat_a, b, vec_x)) > epsilon and iters < 50:
        new_vec_x = [[0] for _ in range(n)]
        norms.append(norm(residuum(mat_a, b, vec_x)))
        iters += 1
        for i in range(0, n):
            sig1 = sum([mat_a[i][j] * new_vec_x[j][0] for j in range(0, i)])
            sig2 = sum([mat_a[i][j] * vec_x[j][0] for j in range(i+1, n)])
            new_vec_x[i][0] = (b[i][0] - sig1 - sig2)/mat_a[i][i]
        for i in range(len(vec_x)):
            vec_x[i][0] = new_vec_x[i][0]

    return vec_x, iters, norms


def zb():
    import time
    m_a, vec_b = construct()

    j_t0 = time.time()
    j_x, j_iters, _ = jacobi(m_a, vec_b)
    j_t = time.time() - j_t0

    gs_t0 = time.time()
    gs_x, gs_iters, _ = gauss_seidel(m_a, vec_b)
    gs_t = time.time() - gs_t0

    print(f"Jacobi: iters={j_iters} residuum={norm(residuum(m_a, vec_b, j_x))} t={j_t}")
    print(f"Gauss-Seidel: iters={gs_iters} residuum={norm(residuum(m_a, vec_b, gs_x))} t={gs_t}")


def zc():
    m_a, vec_b = construct()
    for i in range(len(m_a)):
        m_a[i][i] = 3

    fig, axis = plt.subplots(2)

    j_x, j_iters, j_norms = jacobi(m_a, vec_b)
    gs_x, gs_iters, gs_norms = gauss_seidel(m_a, vec_b)

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


zc()

