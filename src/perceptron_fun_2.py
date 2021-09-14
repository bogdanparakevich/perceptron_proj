import numpy as np


def perceptron_2(x1: tuple, x2: tuple, y_true: tuple, epochs = 30000) -> tuple:
    '''
    function find weights by method gradient descent and nesterov momentum
    return tuple: [w0, w1]
    '''

    n = y_true.shape[0]
    # count n for loss function

    w0 = 1
    w1 = 1
    w2 = 1
    # start random weigth

    alfa = 0.0001
    # learning rate

    momentum = 0.9
    change_0 = 0.001
    change_1 = 0.001
    change_2 = 0.001
    # nesterov momentum


    def model(x1, x2, w0, w1, w2):
        y_pred = w0 + w1 * x1 + w2 * x2
        return y_pred

    for i in range(epochs):
        w0_der = -(1 / n) * 2 * np.sum(y_true - model(x1, x2, w0, w1, w2))
        w1_der = (1 / n) * 2 * np.sum((y_true - model(x1, x2, w0, w1, w2)) * (-x1))
        w2_der = (1 / n) * 2 * np.sum((y_true - model(x1, x2, w0, w1, w2)) * (-x2))
        # partial derivatives
        change_new_w0 = (momentum * change_0) - (alfa * w0_der)
        change_new_w1 = (momentum * change_1) - (alfa * w1_der)
        change_new_w2 = (momentum * change_2) - (alfa * w2_der)
        # nesterov momentum
        w0_new = w0 + change_new_w0
        w1_new = w1 + change_new_w1
        w2_new = w2 + change_new_w2
        w0 = w0_new
        w1 = w1_new
        w2 = w2_new
        change_0 = change_new_w0
        change_1 = change_new_w1
        change_2 = change_new_w2
    return [w0, w1, w2]







