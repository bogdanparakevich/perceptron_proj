import numpy as np

def gen_3D(n_dots: int, x_min: float, x_max: float, w0_model: float, w1_model: float, w2_model: float, M = -0.5, std = 1) -> tuple:
    '''
    @n_dots: number of points
    @x_min: min limit of x
    @x_max: max limit of x
    @w0_model: w0 start weight
    @w1_model: w1 start weight
    @w2_model: w2 start weight
    @M = -0.5: mean (“centre”) of the distribution
    @std = 1: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    function generate x and y data
    return tuples x, y
    return w0_start, w1_start for plotting
    '''
    e1 = np.random.normal(0, 1, (n_dots, ))
    e2 = np.random.normal(1, 1, (n_dots, ))
    e = np.random.normal(-1, 1, (n_dots, ))
    x0 = np.ones(int(n_dots))
    x1 = np.random.randint(x_min, x_max, (n_dots, ))
    x2 = np.random.randint(x_min, x_max, (n_dots, ))
    xx = np.array([x0, x1, x2])
    xx = xx.T
    y_model = w0_model + (w1_model * (x1 + e1)) + (w2_model * (x2 + e2)) + e
    step1 = xx.T.dot(xx)
    step2 = np.linalg.inv(step1)
    step3 = step2.dot(xx.T)
    b = step3.dot(y_model)
    return x1, x2, y_model, w0_model, w1_model, w2_model, b