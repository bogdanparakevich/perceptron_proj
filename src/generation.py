import numpy as np

def gen(n_dots: int, x_min: float, x_max: float, w0_model: float, w1_model: float, M = -0.5, std = 1) -> tuple:
    '''
    @n_dots: number of points
    @x_min: min limit of x
    @x_max: max limit of x
    @w0_model: w0 start weight
    @w1_model: w1 start weight
    @M = -0.5: mean (“centre”) of the distribution
    @std = 1: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    function generate x and y data
    return tuples x, y
    return w0_start, w1_start for plotting
    '''
    e = np.random.normal(M, std, (n_dots, ))
    x0 = np.ones(n_dots)
    x = np.linspace(x_min, x_max, n_dots)
    xx = np.array([x0, x])
    xx = xx.T
    y = w0_model + w1_model * x + e
    step1 = xx.T.dot(xx)
    step2 = np.linalg.inv(step1)
    step3 = step2.dot(xx.T)
    b = step3.dot(y)
    return x, y, w0_model, w1_model, b

