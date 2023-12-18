import numpy as np

def iteration_method(matrix, approximation, eps, max_iter=1e5):
    
    eigenvalue = 0
    eigenvector = approximation.copy()
    iterations = 0
    distance = float('inf')
    
    while distance > eps:
        if iterations == max_iter:
            break
        
        eigenvector_step = matrix @ eigenvector
        norm_step = np.linalg.norm(eigenvector_step)
        
        eigenvalue_step = norm_step / np.linalg.norm(eigenvector)
        eigenvector = eigenvector_step / norm_step
        distance = abs(eigenvalue_step - eigenvalue)
        eigenvalue = eigenvalue_step

    return eigenvalue, eigenvector


def rotation_method(matrix, eps, max_iter=1e5):
    
    A = matrix.copy().astype(float)
    max_element = float('inf')
    n = A.shape[0]
    eigenvectors = np.eye(n)
    iterations = 0
 
    while abs(max_element) > eps:
        if iterations == max_iter:
            break
        
        A_masked = A.copy()
        A_masked[np.tril_indices(n, k=0)] = 0
        i, j = np.unravel_index(np.argmax(abs(A_masked)), A.shape)
        max_element = A[i, j]
        H = np.eye(n)
        
        phi = 0.5 * np.arctan(2 * max_element / (A[i, i] - A[j, j])) if A[i, i] - A[j, j] != 0 else np.pi / 4
        H[i, i] = np.cos(phi)
        H[i, j] = -1 * np.sin(phi)
        H[j, i] = np.sin(phi)
        H[j, j] = np.cos(phi)
 
        A = H.T @ A @ H
        eigenvectors = eigenvectors @ H
        
    return np.diag(A), eigenvectors


matrix = np.array([
    [5, 1, 2],
    [1, 4, 1],
    [2, 1, 3]
], dtype=float)

approximation = np.array([1, 1, 1], dtype=float)

max_eigenvalue, corr_eigenvector = iteration_method(matrix, approximation, 1e-3)
eigenvalues, eigenvectors = rotation_method(matrix, 1e-3)
print(f'Результат работы метода итераций:\n  собственное значение - {max_eigenvalue}\n  собственный вектор - {corr_eigenvector}\n')
print(f'Результат работы метода вращений:\n  собственные значения: {eigenvalues}\n  собственные вектора:\n{eigenvectors}')
