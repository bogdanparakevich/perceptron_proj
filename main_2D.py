from src.generation_2D import gen_2D
from src.perceptron_fun_2D import perceptron_2D
import matplotlib.pyplot as plt


# y_true = w0_model + w1_model * x + e
x, y_true, w0_model, w1_model, b = gen_2D(
    n_dots=400, x_min=0, x_max=1, w0_model=0.2, w1_model=1
)

print("w0_model, w1_model =", w0_model, w1_model)
print("w0_true, w1_true =", perceptron_2D(x, y_true))
print("analytical method test: w0, w1 =", b)


plt.scatter(x, y_true)  # dots
plt.plot(x, w0_model + w1_model * x, "g", label="model")  # model
plt.plot(
    x, perceptron_2D(x, y_true)[0] + perceptron_2D(x, y_true)[1] * x, "r", label="true"
)  # true
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
