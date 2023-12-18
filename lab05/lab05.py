import numpy as np


def global_method(x_points, y_points, point):
    
    n = len(x_points)
    W = np.empty((n, n))
    
    for i in range(n):
        for j in range(n):
            W[i, j] = x_points[i] ** j
    
    coeffs = np.linalg.inv(W) @ y_points
    y_pred = 0
    
    for i, coeff in enumerate(coeffs):
        y_pred = y_pred + coeff * point ** i
    
    return y_pred


def general_case(x_points, y_points, point):

    diagonal_elem_mul = 1
    f_to_D_relation = 0

    for i in range(len(x_points)):
        
        diagonal_elem_mul *= point - x_points[i]
        D = 1

        for j in range(len(x_points)):
            
            D *= (point - x_points[i]) if i == j else (x_points[i] - x_points[j])
        
        f_to_D_relation += y_points[i] / D

    return f_to_D_relation * diagonal_elem_mul


net = [3, 4, 5, 6]
values = [1, 0, 4, 2]
x_star = 4.5

print(f"Результат способа глобальной интерполяции: {global_method(net, values, x_star)}")
print(f"Результат способа кусочно-линейной интерполяции: {general_case(net[1:3], values[1:3], x_star)}")
L1 = general_case(net[0:-1], values[0:-1], x_star)
L2 = general_case(net[1:], values[1:], x_star)
print(f"Результат способа кусочно-параболической интерполяции: {(L1 + L2) / 2}")
