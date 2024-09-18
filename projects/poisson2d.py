import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

def D2(N):
    D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
    D[0, :4] = 2, -5, 4, -1
    D[-1, -4:] = -1, 4, -5, 2
    return D

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        x = np.linspace(0, self.Lx, self.Nx+1)
        y = np.linspace(0, self.Ly, self.Ny+1)
        return np.meshgrid(x, y, indexing='ij')

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = (1./self.dx**2)*D2(self.Nx)
        D2y = (1./self.dy**2)*D2(self.Ny)
        return (sparse.kron(D2x, sparse.eye(self.Ny+1)) +
            sparse.kron(sparse.eye(self.Nx+1), D2y))

    def assemble(self, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        A = self.laplace()
        B = np.ones((self.Nx+1, self.Ny+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        xij, yij = self.create_mesh()
        F = sp.lambdify((x, y), f)(xij, yij)
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = F.ravel()
        b[bnds] = 0
        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        raise NotImplementedError

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = ue.diff(x, 2) + ue.diff(y, 2)
    sol = Poisson2D(1, 1, 30, 30)
    u = sol(f)
    xij, yij = sol.create_mesh()
    assert np.allclose(u, sp.lambdify((x, y), ue)(xij, yij), atol=0.01)

def main():
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = ue.diff(x, 2) + ue.diff(y, 2)
    sol = Poisson2D(1, 1, 30, 30)
    u = sol(f)
    xij, yij = sol.create_mesh()
    #plt.contourf(xij, yij, u)
    plt.contourf(xij, yij, u - sp.lambdify((x,y), ue)(xij, yij))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
