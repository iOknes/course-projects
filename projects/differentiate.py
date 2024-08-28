import numpy as np


def differentiate(u, dt):
    return [(u[idx+1] - val) / dt for idx, val in enumerate(u[:-1])]

def differentiate_vector(u, dt):
    return (u[1:] - u[:-1]) / dt

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
