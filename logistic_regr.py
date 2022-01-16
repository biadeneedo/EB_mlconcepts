"""
Refer to this link for better explanation https://realpython.com/gradient-descent-algorithm-python/
Step per il gradient descent:

1. Definire la funzione da ottimizzare -objective function- (funzione di costo in algoritmi di supervised)
2. Definire il Gradient della funzione (fare la derivata della funzione di costo)
3. Settare dei pesi iniziali dai quali partire
4. Inserire i pesi all'interno della funzione (valorizzazione della funzione di costo)
5.
3. Moltiplicare il gradient per il learning rate
4. Sommare ai pesi il valore del gradiente
"""
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import datasets

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # 01 This is the training function
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X) # this function adds a random variable (intercept) to X

        self.theta = np.zeros(X.shape[1]) # weights

        for i in range(self.num_iter):
            z = np.dot(X, self.theta) # estimation (product between weights and input variables)
            h = self.__sigmoid(z) # estimation transformation (sigmoid transformation of the estimation)
            error = h-y # estimation error
            gradient = np.dot(X.T, error) / y.size
            self.theta -= self.lr * gradient # weights adjustment
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if (self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    # 02 This is the scoring function
    def predict(self, X):
        return self.predict_prob(X).round()


iris = datasets.load_iris()
X = iris.data[:, :2] # only two variables are taken to better visualize the results
y = (iris.target != 0) * 1 # the target variable now have two classes instead of three

model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X, y)
preds = model.predict(X)
(preds == y).mean()
model.theta
