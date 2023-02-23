import numpy as np
import matplotlib.pyplot as plt
from ndiff import ndiff_real,ndiff_imag,ndiff_r,ndiff_theta_by_r
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Global Constants
PI=np.pi

# Shortcuts
conj=np.conjugate
angle=np.angle
sqrt=np.sqrt
exp=np.exp
log=np.log
real=np.real
imag=np.imag

"""
Some nice cyclic colormaps:
    twilight
    hsv
"""

# Functions
def one_over_z(z):
    return 1/z
def jan24_branch2(z):
    return sqrt(z)*sqrt(z-1)*exp(PI*1j)
def jan24(z):
    return sqrt(z)*sqrt(z-1)
def complex_log_branch2(z):
    return log(abs(z)) + 1.j*(angle(z) + 4*PI)
def complex_log(z):
    return log(abs(z)) + 1.j*angle(z)
def pset01_problem04(z):
    return z**2 + (np.real(z)-1)**2 + 1.0j*(np.imag(z)-1)**2
def zbar_squared_over_z(z):
    return np.conjugate(z)**2/z
def julia(z,c=0.1+0.4j):
    """The function that generates the julia set at c"""
    return z**2 + c
def julia_power(z,c=0.1+0.4j,p=2):
    w=z.copy()
    for _ in range(p):
        w=julia(w,c)
    return w
# Speicific wrappers for the above
def julia_c_0_1__0_4j(z):
    return julia(z,c=0.1+0.4j)
def julia_c_0_1__0_4j_power_2(z):
    return julia_power(z,c=0.1+0.4j,p=2)
def julia_c_0_1__0_4j_power_15(z):
    return julia_power(z,c=0.1+0.4j,p=15)

# Plot the phase and amplitude of the function
def plot_func(f,z:np.ndarray,name:str):
    # Function Evaluation
    w=func(Z) # Evaluate function
    # Phase and amplitude
    phase =np.angle(w)
    logamp=np.log10(np.abs(w)) # logarithm of the amplitude
    cutoff=4 # cutoff amplitude range 10^-4 to 10^4
    logamp=np.clip(logamp,a_min=-cutoff,a_max=cutoff) 


    # PLOT
    # ticks, locations and labels
    xticks=np.linspace(xmin,xmax,5)
    yticks=np.linspace(ymin,ymax,5)
    fig, axs = plt.subplots(1,2,figsize=(7,5))
    # Plot the Phase
    plt.subplot(1,2,1)
    plt.pcolormesh(X,Y,phase,cmap="hsv",shading="auto") 
    plt.title("Phase")
    plt.colorbar(orientation='horizontal')
    plt.xticks(xticks)
    plt.yticks(yticks)
    ### Plotting the log amplitude
    # pick a colormap, define level sets and a normalization
    cmap=plt.get_cmap('PuOr')
    vmin=min(logamp.flatten())
    vmax=max(logamp.flatten())
    levels=MaxNLocator(nbins=50).tick_values(vmin,vmax)
    norm=BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    # Plot Amplitude
    plt.subplot(1,2,2)
    plt.pcolormesh(X,Y,logamp,cmap=cmap,shading="auto",norm=norm)
    plt.title("Log10 Amplitude")
    cbar=plt.colorbar(orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='-45')

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.suptitle(f"Evaluating {func.__name__}")

    plt.tight_layout()
    print(f"\nINFO: Saving figure as img/{func.__name__}.png")
    plt.savefig(f"img/{func.__name__}.png",dpi=300)
    plt.show(block=False)
    plt.pause(0.5)


# Plot each of the Cauchy Riemann equation componants using numerical
# differentiation
def plot_cauchy_riemann(f,z,name:str):
    # Unpack complex domain into x,y coords for plotting purposes
    X,Y = np.real(z),np.imag(z)
    # Get the derivatives of f
    dfdx = ndiff_real(f,z)
    dfdy = ndiff_imag(f,z)
    dudx = np.real(dfdx)
    dvdx = np.imag(dfdx)
    dudy = np.real(dfdy)
    dvdy = np.imag(dfdy)

    # color scale parameters
    min_dudx=min(min(dudx.flatten()),min(dvdy.flatten()))
    max_dudx=max(max(dudx.flatten()),max(dvdy.flatten()))
    min_dudy=min(min(dudy.flatten()),-max(dvdx.flatten()))
    max_dudy=max(max(dudy.flatten()),-min(dvdx.flatten()))

    # pick a colormap, define level sets and a normalization
    cmap = plt.get_cmap('PiYG')
    levels1 = MaxNLocator(nbins=15).tick_values(min_dudx, max_dudx)
    norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
    levels2 = MaxNLocator(nbins=15).tick_values(min_dudy, max_dudy)
    norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)

    # TODO: normalize colors
    plt.subplots(3,2,figsize=(7,7))
    plt.suptitle("Cauchy-Rieman Equations")

    plt.subplot(3,2,1)
    plt.title("du/dx")
    plt.pcolormesh(X,Y,dudx,shading="auto",cmap=cmap,norm=norm1)
    plt.colorbar(orientation="horizontal")
    
    plt.subplot(3,2,2)
    plt.title("du/dy")
    plt.pcolormesh(X,Y,dudy,shading="auto",cmap=cmap,norm=norm2)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,3)
    plt.title("-dv/dx")
    plt.pcolormesh(X,Y,-dvdx,shading="auto",cmap=cmap,norm=norm2)
    plt.colorbar(orientation="horizontal")
    
    plt.subplot(3,2,4)
    plt.title("dv/dy")
    plt.pcolormesh(X,Y,dvdy,shading="auto",cmap=cmap,norm=norm1)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,5)
    plt.title("du/dx - dv/dy")
    plt.pcolormesh(X,Y,dudx-dvdy,shading="auto",vmin=min_dudx,vmax=max_dudx,
            cmap=cmap)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,6)
    plt.title("du/dy + dv/dx")
    plt.pcolormesh(X,Y,dudy+dvdx,shading="auto",vmin=min_dudx,vmax=max_dudx,
            cmap=cmap)
    plt.colorbar(orientation="horizontal")


    plt.tight_layout()
    plt.savefig(f"img/{name}_cauchy_riemann.png")
    plt.show(block=False)
    plt.pause(.1)
    return 

# Plot each of the Cauchy Riemann equation componants using numerical
# differentiation, in polar coordinates
def plot_cauchy_riemann_polar(f,z,name:str):
    # Unpack complex domain into x,y coords for plotting purposes
    X,Y = np.real(z),np.imag(z)
    # Get the derivatives of f
    dfdr = ndiff_r(f,z)
    dfdt = ndiff_theta_by_r(f,z)
    dudr = np.real(dfdr)
    dvdr = np.imag(dfdr)
    dudt = np.real(dfdt)
    dvdt = np.imag(dfdt)

    # color scale parameters
    min_dudr=min(min(dudr.flatten()),min(dvdt.flatten()))
    max_dudr=max(max(dudr.flatten()),max(dvdt.flatten()))
    min_dudt=min(min(dudt.flatten()),-max(dvdr.flatten()))
    max_dudt=max(max(dudt.flatten()),-min(dvdr.flatten()))

    # pick a colormap, define level sets and a normalization
    cmap = plt.get_cmap('PiYG')
    levels1 = MaxNLocator(nbins=50).tick_values(min_dudr, max_dudr)
    norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
    levels2 = MaxNLocator(nbins=50).tick_values(min_dudt, max_dudt)
    norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)

    # TODO: normalize colors
    plt.subplots(3,2,figsize=(7,7))
    plt.suptitle("Cauchy-Rieman Equations")

    plt.subplot(3,2,1)
    plt.title("du/dr")
    plt.pcolormesh(X,Y,dudr,shading="auto",cmap=cmap,norm=norm1)
    plt.colorbar(orientation="horizontal")
    
    plt.subplot(3,2,2)
    plt.title("1/r * du/dtheta")
    plt.pcolormesh(X,Y,dudt,shading="auto",cmap=cmap,norm=norm2)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,3)
    plt.title("-dv/dr")
    plt.pcolormesh(X,Y,-dvdr,shading="auto",cmap=cmap,norm=norm2)
    plt.colorbar(orientation="horizontal")
    
    plt.subplot(3,2,4)
    plt.title("1/r * dv/dtheta")
    plt.pcolormesh(X,Y,dvdt,shading="auto",cmap=cmap,norm=norm1)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,5)
    plt.title("du/dr - 1/r * dv/dtheta")
    plt.pcolormesh(X,Y,dudr-dvdt,shading="auto",vmin=min_dudr,vmax=max_dudr,
            cmap=cmap)
    plt.colorbar(orientation="horizontal")

    plt.subplot(3,2,6)
    plt.title("1/r * du/dtheta + dv/dr")
    plt.pcolormesh(X,Y,dudt+dvdr,shading="auto",vmin=min_dudt,vmax=max_dudt,
            cmap=cmap)
    plt.colorbar(orientation="horizontal")


    plt.tight_layout()
    plt.savefig(f"img/{name}_cauchy_riemann_polar.png")
    plt.show()
    return 



### MAIN

# Select function to evaluate
#func=one_over_z
func=jan24
#func=complex_log_branch2
#func=pset01_problem04
#func=zbar_squared_over_z
#func=lambda z:julia_power(z,p=5)
#func=julia_c_0_1__0_4j_power_15
#func=np.conjugate



# Parameters
N=300 # Resolution
xmin,xmax=-2,2 
ymin,ymax=-2,2
x=np.linspace(xmin,xmax,N)
y=np.linspace(ymin,ymax,N)
X,Y=np.meshgrid(x,y)
Z=X+1.0j*Y # Domain

# plot the function as phase and level set amplitude
plot_func(func,Z,func.__name__)

# plot cauchy rieman functions
plot_cauchy_riemann(func,Z,func.__name__)
plot_cauchy_riemann_polar(func,Z,func.__name__)


