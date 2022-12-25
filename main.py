import numpy as np
import pandas as pd
from scipy.stats import norm, genextreme

df = pd.read_excel('xy.xlsx')
X = df['X']
Y = df['Y']

mu, sigma = norm.fit(X)

shape, loc, scale = genextreme.fit(X)

print(shape, loc, scale)

x = np.linspace(genextreme.ppf(0.01, shape, loc, scale),
                genextreme.ppf(0.99, shape, loc, scale), 100)
y = genextreme.pdf(x, shape, loc, scale)

import matplotlib.pyplot as plt
from scipy.stats import probplot

fig, ax = plt.subplots()
probplot(X, plot=ax)
ax.set_title("Probability plot")
plt.show()

# QQ图

import statsmodels.api as sm



sm.qqplot(X, line='s')
plt.show()

fig, ax = plt.subplots()
probplot(Y, plot=ax)
ax.set_title("Probability plot")
plt.show()
