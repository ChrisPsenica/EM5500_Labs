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
[3] Initiate empty grid_dimension, pixel_dimension grid to store solution (initiate as all zeros, this is the reconstruction matrix) 
[4] Calculate xr, yr for all x, y (yr will ultimately be discarded)
[5] Pick closest detector pixel to xr
[6] Add the log(detector value) to the reconstruction matrix
[7] Iterate 4 - 6 over all grid points
[8] Iterate 4 - 7 over all angles
'''

#%---------- Functions ----------
def rotateXY(x , y , phi , mode = str):

    '''
    Description:
    ------------
    This function rotates a set of (x,y) coordinate points through an angle phi. Array inputs must be numpy arrays or an error may occur.

    Inputs:
    -------
    x_inner [array , float] = x-coordinate(s)
    x_outer [array , float] = y-coordinate(s)
    phi [float]             = angle through which to rotate (x,y)
    mode [str]              = whether angle is in radians ('rad') or degrees ('deg')

    Outputs:
    --------
    xr [array , float] = rotated x-coordinate(s)
    yr [array , float] = rotated y-coordinate(s)
    '''

    if mode == "deg":
        phi = phi * np.pi / 180

    return (x * np.cos(phi)) + (y * np.sin(phi)) #, (x * -np.sin(phi)) + (y * np.cos(phi))

#%---------- Process Data ----------
#^ load data
data_set = np.loadtxt("data/data.txt")

#^ [1] scale data
max_detect_val = 3400
indices = np.where(data_set > max_detect_val)
data_set[indices] = max_detect_val
data_set[:] /= 3400

#^ [2] take (standard) logarithm
data_set = -np.log10(data_set)

#%---------- Run Solution ----------
#^ [3] create reconstruction domain 
grid_dimension = 1601 ; pixel_dimension = 200 ; center = 744
xx = np.linspace(1 , grid_dimension , grid_dimension) - center ; yy = np.linspace(1 , grid_dimension , grid_dimension) - center
X , Y = np.meshgrid(xx , yy)
X = X * pixel_dimension ; Y = Y * pixel_dimension
reconstruction_grid = np.zeros((grid_dimension , grid_dimension))

#^ iterate over every angle
total_iters = grid_dimension * grid_dimension * 360
iteration = 1
previous_progress_percent = 0
progress_bar = "["
print("Reconstruction Progress:")
print("------------------------")
print(progress_bar , end = '' , flush = True)

#^ [8] iterate over all angles
for a in range(1 , 360):

    #^ [7] iterate over every pixel
    for i in range(1 , grid_dimension):
        for j in range(1 , grid_dimension): 

            #^ [4] calculate xr
            xr = rotateXY(X[i , j] , Y[i , j] , a , mode = 'deg')

            #^ [5] get closes index to xr
            closest = np.floor(xr / pixel_dimension) + center

            #^ [6] get solution
            if 0 < closest < 1400:
                reconstruction_grid[i , j] += data_set[a , int(closest)]

            #^ check progress
            if ((iteration / total_iters) * 100) - previous_progress_percent >= 1:
                progress_percent = ((iteration / total_iters) * 100)
                progress_bar += "="
                print(f"\r{progress_bar + '>'} {int(np.ceil(progress_percent))}%" , end = '', flush = True)
                previous_progress_percent = progress_percent

            #^ iteration update
            iteration += 1

#%---------- Export Results ----------
np.savetxt('reconstructed_grid.txt' , reconstruction_grid)