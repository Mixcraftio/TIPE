import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2

x=np.linspace(0,50,50)
y=np.zeros(len(x))
for i in range(len(x)):
    y[i]=f(x[i])

plt.plot(x,y,"bo")
plt.axline([0,0],[1,0],c="r")
plt.show()
