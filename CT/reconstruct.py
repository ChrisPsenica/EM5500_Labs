# EM 5500: Computed Tomography
# Chris Psenica
# 12/08/2024

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

#%---------- Lab Description ----------
'''
This lab is about computed tomography. We will post process the given data set to reveal the image. The complete this lab the following steps must be completed:

[1] Scale the data such that the 'straight-thru' portion is 1
[2] Take (standard) logarithm of data set 
[3] Initiate empty Nx, Ny grid to store solution (initiate as all zeros, this is the reconstruction matrix) 
[4] Calculate xr, yr for all x, y (yr will ultimately be discarded)
[5] Pick closest detector pixel to xr
[6] Add the log(detector value) to the reconstruction matrix
[7] Iterate 4 - 6 over all grid points
[8] Iterate 4 - 7 over all angles
'''

#%---------- Functions ----------
def rotateXY(x , y , phi):

    '''
    Description:
    ------------
    This function rotates a set of (x,y) coordinate points through an angle phi. Array inputs must be numpy arrays or an error may occur.

    Inputs:
    -------
    x_inner [array , float] = x-coordinate(s)
    x_outer [array , float] = y-coordinate(s)
    phi [float]             = angle through which to rotate (x,y)

    Outputs:
    --------
    xr [array , float] = rotated x-coordinate(s)
    yr [array , float] = rotated y-coordinate(s)
    '''

    return (x * np.cos(phi)) + (y * np.sin(phi)) , (x * -np.sin(phi)) + (y * np.cos(phi))

#%---------- Process Data ----------
#^ load data
data_set = np.loadtxt("data/data.txt")
shape = np.shape(data_set)
Nx = shape[0] ; Ny = shape[1]

#^ [1] scale data
max_detect_val = 3400
indices = np.where(data_set > max_detect_val)
data_set[indices] = max_detect_val
data_set /= 3400

#^ [2] take (standard) logarithm
data_set = np.log10(data_set)

#%---------- Run Solution ----------
#^ [3] create reconstruction domains 
x = np.linspace(0 , 30 , Nx) ; y = np.linspace(0 , 30 , Ny)
[X , Y] = np.meshgrid(x , y , indexing = 'ij')
reconstruction_grid = np.zeros((Nx , Ny))

#^ create angles to iterate over
phi_array = np.linspace(0 , 2 * np.pi , 360)

#^ iterate over every angle
total_iters = Nx * Ny * int(len(phi_array))
iteration = 1
previous_progress_percent = 0
progress_bar = "|"
print("Simulation Progress:")
print("-------------------")
print(progress_bar , end = '' , flush = True)

#^ [8] iterate over all angles
for phi in phi_array:

    #^ [4] calculate xr
    xr , _ = rotateXY(X , Y , phi)

    #^ [7] iterate over every pixel
    for i in range(Nx):
        for j in range(Ny):

            #^ [5] get closes index to xr
            closest = np.argmin(np.abs(x - xr[i , j]))

            #^ [6] get solution
            reconstruction_grid[i , j] += data_set[closest , j]

            #^ check progress
            if ((iteration / total_iters) * 100) - previous_progress_percent >= 1:

                progress_percent = ((iteration / total_iters) * 100)
                progress_bar += "="
                print(f"\r{progress_bar + '>'} {int(progress_percent)}%" , end = '', flush = True)
                previous_progress_percent = progress_percent

            #^ iteration update
            iteration += 1

#%---------- Plot Results ----------
plt.figure(figsize = (10 , 10))
plt.imshow(reconstruction_grid , aspect = 'auto') 
plt.title('Reconstruction' , color = "blue") 
plt.colorbar()
plt.show() 