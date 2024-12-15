# EM 5500: Computed Tomography - Plot reconstructed data
# Chris Psenica
# 12/14/2024

#%---------- Color Key ----------
# & (neon) pink
# ! red
# ^ pink
# ? blue
# ~ green
# % orange
# | purple

#%---------- imports ----------
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

#%---------- Import Data ----------
reconstruction_grid = np.loadtxt("data/reconstructed_grid.txt") / 360

#%---------- Plot & Save Initial Results ----------
plt.figure(figsize = (10 , 10))
plt.imshow(reconstruction_grid , aspect = 'auto' , cmap = 'jet') 
plt.title('Initial Reconstruction' , color = "blue") 
plt.colorbar()
plt.show() 

#%---------- Deblur Results ----------
#^ parameters
dx = 200 ; nx = 1601
eps = 0.00001 
kx = 2 * np.pi * np.fft.fftfreq(nx , d = dx)
kx = np.fft.fftshift(kx)

#^ Calculate M_b(k) using 2D FFT
M_bk = fftshift(fft2(reconstruction_grid))

#^ subtract 2π/dx 
kx[nx//2:] -= (2 * np.pi) / dx
Kx , Ky = np.meshgrid(kx , kx)
rho = np.sqrt(Kx ** 2. + Ky ** 2.)
rho[rho == 0] = 1e-6     

#^ filter
high_resolution = np.pi * rho * np.exp(-eps * rho)
high_reconstruct = np.real(ifft2(ifftshift(M_bk * high_resolution)))

#^ plot
plt.figure(figsize = (10 , 10))
plt.imshow(reconstruction_grid , aspect = 'auto' , cmap = 'jet') 
plt.title('Back Projection' , color = "blue") 
plt.colorbar()
plt.show() 

plt.figure(figsize = (10 , 10))
plt.imshow(high_reconstruct , aspect = 'auto' , cmap = 'jet') 
plt.title('Deblurred Reconstruction' , color = "blue") 
plt.colorbar()
plt.show() 