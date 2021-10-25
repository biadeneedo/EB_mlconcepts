#  GRADIENT DESCENT FOR SINGLE VARIABLE OPTIMIZATION
def gradient_descent(
    function, gradient, start, learn_rate, n_iter=50, tolerance=1e-06
):
    vector = start
    for _ in range(n_iter):
        print(function(vector))
        diff = -learn_rate * gradient(vector)
        print('diff:', diff)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        print('vector:', vector)
    return vector

gradient_descent(
    function=lambda v: v ** 2, gradient=lambda v: 2 * v, start=10.0, learn_rate=0.2
)


# GRADIENT DESCENT FOR TWO VARIABLES OPTIMIZATION
# Used for linear regression algorithms
# Refer to this link for better explanation https://realpython.com/gradient-descent-algorithm-python/
import numpy as np
def gradient_descent(
    cost_function, gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06
):
    vector = start
    for i in range(n_iter):
        print(i)
        print('cost_funtion', cost_function(vector[0], vector[1]))
        diff = -learn_rate * np.array(gradient(x, y, vector))
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        print('vector:', vector)
    return vector

def ssr_gradient(x, y, b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])
cost_function = lambda m, b: (y - m*x**2 - b)/ 2*(len(x)) #This is the square residuals formula coming from a simple regression
gradient_descent(cost_function, ssr_gradient, x, y, start=[0.5, 0.5], learn_rate=0.0008, n_iter=100_000)




