import numpy as np

'''Функция подбора интервала для поиска корней Методом Половинного Деления'''


def approximate_interval(_coeffs, _type='viete', _spread=1e4):
    interval = None

    if _type == 'viete':
        coeffs = np.array(_coeffs)
        max_interval = 1 + np.max(coeffs[1:] / coeffs[0])
        interval = [-max_interval, max_interval]

    if _type == 'random':
        interval = sorted(np.random.randint(-_spread, _spread, 2).tolist())

    return interval


'''Функция подбора стратовой точки для Метода Простых Итераций'''


def approximate_point(_bias, _spread=1e4):
    return _bias + np.random.randint(-_spread, _spread, 1)[0]


'''Функция создания полинома'''


def make_polynom(_coeffs, _roots=[]):
    poly = np.poly1d(_coeffs) / (1 if _roots == [] else np.poly1d(_roots, True))
    if isinstance(poly, tuple):
        poly = poly[0]

    return poly


'''Функция поиска корня на заданном интервале Методом Половинного Деления'''


def dichotomy(_coeffs, _interval, _eps, _roots=[]):
    poly = make_polynom(_coeffs, _roots)
    x_left = min(_interval)
    x_right = max(_interval)
    x_mean = np.mean([x_left, x_right])

    while x_right - x_left > _eps:

        if poly(x_left) * poly(x_right) > 0:
            return None

        if poly(x_left) * poly(x_mean) > 0:
            x_left = x_mean
        else:
            x_right = x_mean

        x_mean = np.mean([x_left, x_right])

    return x_mean


'''Функция поиска всех корней на интервале'''


def find_roots(_coeffs, _iterations=100, _eps=0.01, _spread=1e4):
    def step():
        interval = approximate_interval(coeffs, _type='random', _spread=_spread)
        root = dichotomy(coeffs, interval, _eps, roots)
        return interval, root

    coeffs = _coeffs
    roots = []
    interval, root = step()

    for _ in range(_iterations):
        if root is None:
            interval, root = step()
            continue

        root = np.round(root)
        while np.isclose(make_polynom(_coeffs, _roots=roots)(root), 0., atol=_eps):
            roots.append(root)
        coeffs = make_polynom(_coeffs, _roots=roots).coeffs

        interval, root = step()

    return roots


'''Функция поиска корня Методом Простых Итераций'''


def simple_iterations(_start_point, _bias, _eps, _iterations=100):
    step = lambda x: (_bias / x + x) / 2
    func = lambda x: x - np.sqrt(_bias)
    x_prev = _start_point

    for i in range(_iterations):
        if func(x_prev) <= _eps:
            return x_prev
        x_prev = step(x_prev)

    return x_prev if np.isclose(func(x_prev), 0.0, atol=_eps) else None


'''Пример работы 1ого задания'''
polynom = np.poly1d([2, 3, 3, 7], True).coeffs
print(f"Problem #1: {find_roots(polynom, _iterations=1000, _spread=10)}")

'''Пример работы 2ого задания'''
a = 1001
print(f"Problem #2: {simple_iterations(approximate_point(a, _spread=a / 2), a, 0.01)}")
