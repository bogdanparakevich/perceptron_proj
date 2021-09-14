from src.generation_3D import gen_3D
from src.perceptron_fun_3D import perceptron_3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


x1, x2, y_model, w0_model, w1_model, w2_model, b = gen_3D(
    n_dots=100, x_min=1, x_max=10, w0_model=0.2, w1_model=1, w2_model=2
)

print("w0_model, w1_model, w2_model =", w0_model, w1_model, w2_model)
print("w0_true, w1_true, w2_true =", perceptron_3D(x1, x2, y_model))
print("analytical method test: w0, w1, w2 =", b)
y_true = (
    perceptron_3D(x1, x2, y_model)[0]
    + perceptron_3D(x1, x2, y_model)[1] * x1
    + perceptron_3D(x1, x2, y_model)[2] * x2
)

X, Y = np.meshgrid(x1, x2)

w0 = perceptron_3D(x1, x2, y_model)[0]
w1 = perceptron_3D(x1, x2, y_model)[1]
w2 = perceptron_3D(x1, x2, y_model)[2]


def f(x1, x2):
    return w0 + w1 * x1 + w2 * x2


Z = f(X, Y)

ax = plt.axes(projection="3d")

xdata = x1
ydata = x2
zdata = y_model
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Reds")

ax.scatter3D(xdata, ydata, y_true, c=y_true, cmap="Greens")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

surf = ax.plot_surface(X, Y, Z, cmap="Blues")

plt.show()
