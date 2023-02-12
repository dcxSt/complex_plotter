import numpy as np

### Numerical derivatives
def ndiff(f,x,idx):
    """Take the partial derivative of f at x wrt to the idx'th argument.

    f : function
        A multivariate function
    x : np.ndarray
        A vector argument, where we evaluate the derivaitve at. The 
        shape of this array is (number_of_variables, anything, ...)
    idx : int
        The index of the direction along which we evaluate the partial
        deriv
    """
    # Select some optimal step sizes
    # We approximate roundoff error epsilon=10.0e-16
    # Highest order term in 2 point derivative TS expansion is o(dx**3)
    # Optimal step size is about 2*sqrt(epsilon) = 2.0e-08
    # (ballpark back of envolope estimate, assume f order 1)
    dx = 2.0e-08
    step_h = np.zeros(x.shape)
    step_h[idx] = dx
    # Return the numerical partial derivative of f wrt it's argument at
    # idx, at x
    return (f(x+step_h) - f(x-step_h))/(2*dx)

def ndiff_real(f,z:np.ndarray):
    """Take partial derivative of complex function f wrt real part at z

    f : function
        A complex function C -> C
    z : np.ndarray
        A vector argument, where we evaluate the derivative at. 
    """
    dx = 2.0e-08
    return (f(z+dx) - f(z-dx))/(2*dx)

def ndiff_imag(f,z:np.ndarray):
    """Take partial derivative of complex function wrt to imag part"""
    dy = 2.0e-08
    return (f(z+1.j*dy) - f(z-1.j*dy))/(2*dy)

def ndiff_r(f,z:np.ndarray):
    """Partial derivative of complex function wrt radial direction"""
    delta = 2.0e-08
    dr = delta*z/np.abs(z) # pointwise division
    return (f(z+dr) - f(z-dr))/(2*delta)

def ndiff_theta_by_r(f,z:np.ndarray):
    delta = 2.0e-08
    z_rot=-np.imag(z)+1.0j*np.real(z)# rotate z by 90 degrees counter-clockwise
    dtheta = delta*z_rot/np.abs(z_rot)
    return (f(z+dtheta) - f(z-dtheta))/(2*delta)



if __name__=="__main__":
    # Create grid of complex points to evaluate function at
    N=5 # number of grid-points is N^2
    x=np.linspace(-2,2,N)
    y=np.linspace(-2,2,N)
    X,Y=np.meshgrid(x,y)
    Z=X+1.j*Y # Domain of evaluation
    print("INFO: Testing ndiff_real")
    def func(z):
        return np.real(z)
    out=func(Z)
    print("\nFunction is just real part")
    print(out)
    print("\nNdiff real, should be grid of ones")
    print(ndiff_real(func,Z))
    print("\nNdiff imag, should be grid of zeros")
    print(ndiff_imag(func,Z))





