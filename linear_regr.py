"""
Refer to this link for better explanation https://realpython.com/gradient-descent-algorithm-python/
Step per il gradient descent:

1. Definire la funzione da ottimizzare -objective function- (funzione di costo in algoritmi di supervised)
2. Definire il Gradient della funzione (fare la derivata della funzione di costo)
3. Settare dei pesi iniziali dai quali partire
4. Inserire i pesi all'interno della funzione (valorizzazione della funzione di costo)
3. Moltiplicare il gradient per il learning rate
4. Sommare ai pesi il valore del gradiente
"""

#  GRADIENT DESCENT FOR SINGLE-VARIABLE OPTIMIZATION
#  The function is x^2
#  The gradient is 2x
def gradient_descent(
    function, gradient, start, learn_rate, n_iter=50, tolerance=1e-06
):
    vector = start
    for _ in range(n_iter):
        print('function:', function(vector))
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


# GRADIENT DESCENT FOR TWO-VARIABLE OPTIMIZATION
# Used for linear regression algorithms y=mx+b
# The function is the Sum of Squared Residuals SSR=Σᵢ(𝑦ᵢ − 𝑏 − m𝑥ᵢ)² /(2𝑛)
# The gradient are: Σᵢ(𝑏 + m𝑥ᵢ − 𝑦ᵢ) *(1/𝑛)  &  Σᵢ(𝑏 + m𝑥ᵢ − 𝑦ᵢ) *𝑥ᵢ *(1/𝑛)
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




# GRADIENT DESCENT FOR MULTIPLE-VARIABLE OPTIMIZATION - LINEAR REGRESSION
# Used for this linear regression algorithms is y= m2x1+ m2x2 (+b?)
# The function is the mean squared error (MSE)
import numpy as np
def gradient_descent(
    cost_function, gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06
):
    vector = start
    for i in range(n_iter):
        print(i)
        print('cost_funtion', cost_function(x, y, vector))
        for c in range(x.shape[1]):
            diff = - learn_rate * np.array(gradient(x, y, vector, c))
            vector[c] += diff
        if np.all(np.abs(diff) <= tolerance):
            break
        print('vector:', vector)
    return vector

def cost_function(x, y, theta):
    estimation = np.sum(theta * x, axis = 1)
    mse = np.mean((y - estimation)**2)
    return mse

def ssr_gradient(x, y, theta, c):
    estimation = np.sum(theta * x, axis = 1)
    gradient = np.mean(-2*x[:,c] * (y - estimation))
    return gradient

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data[:, :2] # only two variables are taken to better visualize the results
y = (iris.target != 0) * 1 # the target variable now have two classes instead of three
theta = np.array([0.4, 8.5])
gradient_descent(cost_function, ssr_gradient, x, y, start=theta, learn_rate=0.008, n_iter=10000)
