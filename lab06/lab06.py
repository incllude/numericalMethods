from math import factorial
import numpy as np


def partitioned_difference(x_points, y_points): 
    
    if len(y_points) == 2:
        return (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
    
    return (partitioned_difference(x_points[1:], y_points[1:]) - partitioned_difference(x_points[:-1], y_points[:-1])) / (x_points[-1] - x_points[0])
    

def finite_difference(y_points, index, power):
    
    if power == 1:
        return y_points[index + 1] - y_points[index]
    
    return finite_difference(y_points, index + 1, power - 1) - finite_difference(y_points, index, power - 1)
        

def newton_polynom_nonuniform(x_points, y_points, point, reverse=False):
    
    if reverse:
        x_points = x_points[::-1]
        y_points = y_points[::-1]
    
    y_pred = y_points[0]
        
    for i in range(1, len(x_points)):
        
        x_factorial = 1
        
        for j in range(i):
            
            x_factorial *= point - x_points[j]

        y_pred += partitioned_difference(x_points[:i+1], y_points[:i+1]) * x_factorial

    return y_pred


def newton_polynom_uniform(x_points, y_points, point, reverse=False):

    if reverse:
        x_points = x_points[::-1]
        y_points = y_points[::-1]
    
    h = x_points[1] - x_points[0]
    
    for i in range(len(x_points)):
        if i == 0:
            y_pred = y_points[0]
            q0 = (point - x_points[0]) / h
            q = 1
            continue

        q *= q0 - i + 1
        y_pred += finite_difference(y_points, 0, i) * q / factorial(i)

    return y_pred


net    = [0, 1, 2, 3]
values = [1, 2, 4, 1]
x_star = 1.5

print(f"Многочлен Ньюютона N_3 для неравномерной сетки: {newton_polynom_nonuniform(net, values, x_star)}")
print(f"Многочлен Ньюютона N_3^I для равномерной сетки: {newton_polynom_uniform(net, values, x_star)}")
print(f"Многочлен Ньюютона N_3^II для равномерной сетки: {newton_polynom_uniform(net, values, x_star, reverse=True)}")
print(f"Многочлен Ньюютона N_4^I для равномерной сетки: {newton_polynom_uniform(net + [4], values + [0], x_star)}") 
