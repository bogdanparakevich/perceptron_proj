from generation import gen
from perceptron_fun import perceptron
import matplotlib.pyplot as plt


# y_true = w0_start + w1_start * x + e
x, y_true, w0_start, w1_start = gen(n_dots = 50, x_min = 0, x_max = 1, w0_start = 0.2, w1_start = 1, M = -0.5, std = 1)

print('w0_true and w1_true =', perceptron(x, y_true))
print('x =', x.round(3))
print('y_true =', y_true.round(3))


plt.scatter(x, y_true) # dots
plt.plot(x, w0_start + w1_start * x, 'g', label='model') # model
plt.plot(x, perceptron(x, y_true)[0] + perceptron(x, y_true)[1] * x, 'r', label='true') # true 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



