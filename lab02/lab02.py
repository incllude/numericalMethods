import numpy as np

def derivative(polynom):
    
    derivatived_coeffs = []
    
    for i, coeff in enumerate(polynom.coeffs[::-1][1:]):

        derivatived_coeffs.append(coeff * (i + 1))
        
    return np.poly1d(derivatived_coeffs[::-1])


def approximate_derivative(polynom, a, b):
    
    return (polynom(a) - polynom(b)) / abs(b - a)


def newton_method(polynom, polynom_l, x0, eps, lr=1.0, max_iter=10000):
    
    x_step = x0
    iterations = 0
    distance = float('inf')
    
    while distance > eps:
        if iterations == max_iter:
            return None
        
        x = x_step
        x_step = x - lr * (polynom(x) / polynom_l(x))
        distance = abs(x_step - x)
        iterations += 1

    return x


def newton_light_method(polynom, derivative, x0, eps, max_iter=10000):
    
    x_step = x0
    iterations = 0
    distance = float('inf')
    
    while distance > eps:
        if iterations == max_iter:
            return None
        
        x = x_step
        x_step = x - polynom(x) / derivative
        distance = abs(x_step - x)
        iterations += 1
    
    return x


def secants_method(polynom, x0, delta, eps, max_iter=10000):
    
    x_step = x0
    x = x_step - delta
    iterations = 0
    distance = float('inf')
    
    while distance > eps:
        if iterations == max_iter:
            return None
        
        derivative_ = approximate_derivative(polynom, x_step, x)
        x = x_step
        x_step = x - polynom(x) / derivative_
        distance = abs(x_step - x)
        iterations += 1
    
    return x


def chords_method(polynom, a, b, eps, max_iter=10000):
    
    right, left = a, b
    x = right
    iterations = 0
    distance = float('inf')
    
    while distance > eps:
        if iterations == max_iter:
            return None
        
        right = x
        x = right - (polynom(right) * (left - right)) / (polynom(left) - polynom(right))
        distance = abs(x - right)
        iterations += 1
    
    return x


polynom = np.poly1d([4, 0, 10, -10])
start_point = 2
eps = 1e-6
print(polynom)
print(f"Результат метода Ньютона-Бройдена: x = {newton_method(polynom, derivative(polynom), start_point, eps, lr=0.3)}")
print(f"Результат метода Ньютона: x = {newton_method(polynom, derivative(polynom), start_point, eps, 1)}", )
print(f"Результат прощенного метода Ньютона: x = {newton_light_method(polynom, derivative(polynom)(start_point), start_point, eps)}")
print(f"Результат метода секущих: x = {secants_method(polynom, start_point, 0.0001, eps)}")
print(f"Результат метода хорд: x = {chords_method(polynom, start_point - 1, start_point + 1, eps)}")
