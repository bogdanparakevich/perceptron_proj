from src.generation import gen
from src.perceptron_fun import perceptron
import matplotlib.pyplot as plt


# y_true = w0_model + w1_model * x + e
x, y_true, w0_model, w1_model,b = gen(n_dots = 50, x_min = 0, x_max = 1, w0_model = 0.2, w1_model = 1, M = -0.5, std = 1)

print('w0_model, w1_model =', w0_model, w1_model)
print('w0_true, w1_true =', perceptron(x, y_true))
print('analytical method test: w0, w1 =', b)


plt.scatter(x, y_true) # dots
plt.plot(x, w0_model + w1_model * x, 'g', label='model') # model
plt.plot(x, perceptron(x, y_true)[0] + perceptron(x, y_true)[1] * x, 'r', label='true') # true 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



