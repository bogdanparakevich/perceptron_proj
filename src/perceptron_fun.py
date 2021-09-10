import numpy as np


def perceptron(x: tuple, y_true: tuple) -> tuple:
    '''
    function find weights by method gradient descent and nesterov momentum
    return tuple: [w0, w1]
    '''

    n = y_true.shape[0]
    # count n for loss function

    w0 = 1
    w1 = 1
    # start random weigth

    alfa = 0.0001
    # learning rate

    momentum = 0.9
    change_0 = 0.001
    change_1 = 0.001
    # nesterov momentum

    delta = 1e-7
    epochs = 100000

    def model(x, w0, w1):
        y_pred = w0 + w1 * x
        return y_pred

    for i in range(epochs):
        w0_der = -(1 / n) * 2 * np.sum(y_true - model(x, w0, w1))
        w1_der = (1 / n) * 2 * np.sum((y_true - model(x, w0, w1)) * (-x))
        # partial derivatives
        change_new_w0 = (momentum * change_0) - (alfa * w0_der)
        change_new_w1 = (momentum * change_1) - (alfa * w1_der)
        # nesterov momentum
        w0_new = w0 + change_new_w0
        w1_new = w1 + change_new_w1
        if (abs(w0_new - w0) < delta) and (abs(w1_new - w1) < delta):
            break
        w0 = w0_new
        w1 = w1_new
        change_0 = change_new_w0
        change_1 = change_new_w1
    return [w0.round(3) , w1.round(3)]







