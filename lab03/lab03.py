from math import sqrt
import numpy as np


def solve_system(matrix):
    
    solution = matrix.copy()
    n = len(matrix)
    
    for i in range(n - 1, -1, -1):
        
        value = solution[i, -1]
        
        for upper_i in range(i):
            
            solution[upper_i, -1] -= solution[upper_i, i] * value
            solution[upper_i, i] = 0
            
    return solution[:, -1]


def matrix_norm(matrix):
    
    norm = 0
    
    for vector in matrix:
        norm += vector_norm(vector) ** 2
    
    return sqrt(norm)


def vector_norm(vector):
    
    return sqrt(np.sum(vector * vector))
    
    
def gauss_method(matrix):
    
    transformed = matrix.copy()
    n = len(matrix)
    
    for i in range(n):
        
        diag_elem = transformed[i, i]
        transformed[i] /= diag_elem
        
        for lower_i in range(i + 1, n):
            
             transformed[lower_i] -= transformed[lower_i, i] * transformed[i]

    solution = solve_system(transformed)

    return solution


def gauss_jordan_method(matrix):
    
    transformed = matrix.copy()
    n = len(matrix)
    
    for k in range(n):
        
        previous = transformed.copy()
        
        for i in range(k, n):
            for j in range(n + 1):
                
                if   i == k:
                    transformed[i, j] /= previous[k, k]
                elif j <= k:
                    transformed[i, j] = 0
                else:
                    transformed[i, j] -= previous[k, j] * previous[i, k] / previous[k, k]
                    
    solution = solve_system(transformed)

    return solution


def lu_method(matrix, answer):
    
    n = matrix.shape[0]
    l_matrix = np.zeros((n, n))
    u_matrix = np.eye(n)
    
    def scalar_dot(a, b, m):
        
        return a[:m] @ b[:m]
    
    for k in range(n):
        for j in range(k, n):
            l_matrix[j, k] = matrix[j, k] - scalar_dot(l_matrix[j], u_matrix[:, k], k)
        for i in range(k + 1, n):
            u_matrix[k, i] = (matrix[k, i] - scalar_dot(l_matrix[k], u_matrix[:, i], i)) / l_matrix[k, k]

    solution = np.linalg.inv(l_matrix @ u_matrix) @ answer

    return solution


def make_alpha_beta(matrix):
    
    alpha, beta = matrix[:, :-1].copy(), matrix[:, -1].copy()
    max_values = np.diag(matrix)
    alpha = -1 * (alpha.T / max_values).T
    np.fill_diagonal(alpha, 0)
    beta = beta / max_values
    
    return alpha, beta


def get_max_on_diagonal(matrix):
    
    transformed = matrix.copy()
    
    for i in range(matrix.shape[0]):
        
        max_row = np.argmax(abs(matrix[i:, i])) + i
        if max_row != i:
            transformed[[i, max_row]] = transformed[[max_row, i]]

    return transformed

def simple_iterations_method(matrix, eps, max_iter=1e5):
    
    transformed = matrix.copy()
    transformed = get_max_on_diagonal(transformed)
    alpha, beta = make_alpha_beta(transformed)
    
    x_step = beta.copy()
    distance = float('inf')
    iterations = 0
    
    while distance > eps and iterations != max_iter:
        
        x = x_step.copy()
        x_step = alpha @ x + beta
        distance = matrix_norm(alpha) / (1 - matrix_norm(alpha)) * vector_norm(x_step - x)
        iterations += 1
    
    return x


def zeydel_method(matrix, eps, max_iter=1e5):
    
    n = len(matrix)
    transformed = matrix.copy()
    transformed = get_max_on_diagonal(transformed)
    alpha, beta = make_alpha_beta(transformed)
    
    x_step = beta.copy()
    distance = float('inf')
    iterations = 0
    
    while distance > eps and iterations != max_iter:
        
        idx = iterations % n
        x = x_step.copy()
        x_step[idx] = alpha[idx, :] @ x + beta[idx]
        distance = matrix_norm(alpha) / (1 - matrix_norm(alpha)) * vector_norm(x_step - x)
        iterations += 1
    
    return x


matrix_1 = np.array([
    [1, -1,  1, -4, -2],
    [2,  1, -5,  1,  2],
    [8, -1, -1,  2, 11],
    [1,  6, -2, -2, -7]
], dtype=float)

matrix_2 = np.array([
    [1,  4,  3, 10],
    [2,  1, -1, -1],
    [3, -1,  1, 11]
], dtype=float)

matrix_3 = np.array([
    [2, 1, 4, 16],
    [3, 2, 1, 10],
    [1, 3, 3, 16]
], dtype=float)

matrix_4 = np.array([
    [ 2,  2, 10, 14],
    [10,  1,  1, 12],
    [ 2, 10,  1, 13]
], dtype=float)


print(f"Результат метода единичного деления: x = {gauss_method(matrix_1)}")
print(f"Результат метода исключения: x = {gauss_jordan_method(matrix_2)}")
print(f"Результат метода LU разложение: x = {lu_method(matrix_3[:, :-1], matrix_3[:, -1])}")
print(f"Результат метода простых итераций: x = {simple_iterations_method(matrix_4, 1e-9, 1e9)}")
print(f"Результат метода Зейделя: x = {zeydel_method(matrix_4, 1e-9, 1e9)}")
