import numpy as np


def mesh_function(f, t):
    return f(t)

def func(t):
    result = np.full_like(t, np.nan, dtype="float64")
    result[(0 <= t) * (t <= 3)] = np.exp(-t)[(0 <= t) * (t <= 3)]
    result[(3 < t) * (t <= 4)] = np.exp(-3 * t)[(3 < t) * (t <= 4)]
    return result

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
