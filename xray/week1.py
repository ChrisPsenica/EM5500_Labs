# XRay Lab
# Chris Psenica

#---------- imports ----------
import numpy as np
import matplotlib.pyplot as plt
from mat_data import tungsten_p , copper_p , tungsten_c , copper_c , air_c , air_p

#================================================================================
# Part 1 - Setup
#================================================================================
#---------- Define Materials / Initiate Material Matrices ----------
'''
Select a material a contrasts source of your choice. We will have a voxel grid which is made up of both of these
materials. Represent this a 3D matrix indexed by (x , y , z). We will set an array up of a 3x3 cube made up of 
smaller cubes where each small cube is representative of one of the two materials you have chosen.
'''

# Material 1: tungsten -> coolest metal
# Material 2: copper   -> reminds me of pipe case

'''
Select numbers to represent each material as well as get their densities.
'''

# 0 -> vacuum       | density = 0    
# 1 -> tungsten     | density = 19.3      g/cm^3
# 2 -> copper       | density = 8.96      g/cm^3
# 3 -> air          | density = 1.225e-3  g/cm^3
density = [0 , 19.3 , 8.96 , 1.225e-3]    # array of density values [air , tungsten , copper] in kg/m^3
z_detect = 0.14         # z location of the detector
grid_points = 201    # resolution (pixel density)

lower_bnd = -5
upper_bnd = 5

nx = grid_points   # number of points in x direction for voxel grid
ny = grid_points   # number of points in y direction for voxel grid
nz = grid_points   # number of points in z direction for voxel grid

mat_array = np.zeros((nx , ny , nz) , dtype = np.uint32)   # voxel grid (material array)
x = np.linspace(lower_bnd , upper_bnd , nx)                # x range
y = np.linspace(lower_bnd , upper_bnd , ny)                # y range
z = np.linspace(lower_bnd, upper_bnd , nz)                 # z range

(X , Y , Z) = np.meshgrid(x , y , z , indexing = "ij")    # initiate meshgrid for 3D x , y , z with center element at (x , y , z) = (0 , 0 , 0)
r_outer = 3                                               # outer radius of tungsten sphere is 3
r_inner = 3                                               # outer radius of copper sphere is 1

mat_array[(X ** 2. + Y ** 2. + Z ** 2. < r_outer ** 2.) & (X > 0)] = 2      # makes a tungsten half sphere at the center of voxel grid
mat_array[(X ** 2. + Y ** 2. + Z ** 2. < r_inner ** 2.) & (X < 0)] = 3      # makes a copper half sphere at the center of voxel grid

#---------- Model Photon Scattering ----------
'''
I = I_0 * e ^ (-mew * dz)

I   = Output intensity  -> photons/sec
I_0 = Input intensity -> photons/sec
mew = Absorption coefficient = alpha * density
dz  = Change in z

I -> Represents the number of photons not scattered over dz

We want photoelectric absorption and compton scattering from XCOM. We must retrieve this data. This will give us:

I_total = I_0 * e ^(-(mew_p + mew_c) * dz)

mew_p = mew for photoelectric
mew_c = mew for compton

Our total absorption then becomes:

Absorb_total = I_0 - I_0 * e ^(-(mew_p + mew_c) * dz)

Finally we have:

Total_photo_absorb = I_0 - I_0 * e ^ (-mew_p * dz)
not_photo_absorb = I_0 * e ^ (-mew_p * dz)

Total_comp = I_0 - I_0 * e ^ (-mew_c * dz)
not_comp = I_0 * e ^ (-mew_c * dz)
'''

#---------- mew Function ----------
def get_mew(energy , table , density):
    '''
    This function takes in a data table and the corresponding material's density and calculates the absorption coefficient

    Inputs:
    -------
    energy  = photon energy
    table   = material data array (tables located in mat_data.py)
    density = material density (kg/m^3)

    Outputs:
    --------
    mew = absorption coefficients
    '''
    index = (np.abs(table[: , 0] - energy)).argmin()
    mew = table[index , 1] * density #* 1e2

    return mew

#---------- Calculate mew ----------                                 # mew for the vacuum is zero
photon_energy = 4                                                 # energy of the photon in MeV
mew_copper_c = get_mew(photon_energy , copper_c , density[2])        # mew -> coefficient of absorption for copper for compton scattering
mew_copper_p = get_mew(photon_energy , copper_p , density[2])        # mew -> coefficient of absorption for copper for photoelectric absorption
mew_tungsten_c = get_mew(photon_energy , tungsten_c , density[1])    # mew -> coefficient of absorption for tungsten for compton scattering
mew_tungsten_p = get_mew(photon_energy , tungsten_p , density[1])    # mew -> coefficient of absorption for tungsten for photoelectric absorption
mew_air_c = get_mew(photon_energy , air_c , density[3])              # mew -> coefficient of absorption for air for compton scattering
mew_air_p = get_mew(photon_energy , air_p , density[3])              # mew -> coefficient of absorption for air for photoelectric absorption

#---------- Detector ----------
(X_2d , Y_2d) = np.meshgrid(x , y , indexing = "ij")        # meshgrid for detector array
detector_array = np.zeros(X_2d.shape , dtype = np.int32)    # detector array
dz = z[1] - z[0]                                            # change in z through voxel grid

#================================================================================
# Part 2 - Compute
#================================================================================
#---------- Photoelectric Absorption ----------
I_0 = 1e8    # initial photons/sec value

def photons(I_i , pixel_val):
    '''
    inputs:
    -------
    I_i       = Incoming photon/sec value on the pixel
    pixel_val = Identification number of what material the pixel represents

    outputs:
    --------
    I   = Total intensity
    '''
    
    if pixel_val == 0: # vacuum
        mew_p = mew_c = 0

    elif pixel_val == 1: # air
        mew_p = mew_air_p
        mew_c = mew_air_c

    elif pixel_val == 2: # tungsten
        mew_p = mew_tungsten_p
        mew_c = mew_tungsten_c

    elif pixel_val == 3: # copper
        mew_p = mew_copper_p
        mew_c = mew_copper_c

    I = I_i * np.exp(-(mew_p + mew_c) * dz)    # total intensity
    not_comp = I_i * np.exp(-mew_c * dz)

    return not_comp , I

solution_array = np.zeros((nx , ny))
I_array = np.zeros((nx , ny))
I_array[:] = I_0
z = 0
iteration = 0

#---------- Iterate Over Each Layer Of Voxel Grid ----------
for k in range(nz):         # iterate in z-direction
    for i in range(nx):   
        for j in range(ny): # iterate over face of voxel grid (x-y plane)
            print("Iteration:" , iteration)
            iteration += 1
            solution_array[i , j] , I_array[i , j] = photons(I_array[i , j] , int(mat_array[i , j , k]))

z_air = 0
while z_air <= z_detect:
    for i in range(nx):   
        for j in range(ny): # iterate over face of voxel grid (x-y plane)

            solution_array[i , j] , I_array[i , j] = photons(I_array[i , j] , 1)

    z_air += dz

#---------- Plot Photons ----------
c = plt.imshow(solution_array.T , cmap ='rainbow', vmin = 0, vmax = 1.2 * I_0, extent =[x.min(), x.max(), y.min(), y.max()], interpolation ='nearest', origin ='lower') 
plt.colorbar(c) 
plt.xlabel("X Location On Sensor")
plt.ylabel("Y Location On Sensor")
plt.title("Sensor Readout Of The Number Of Photons Per Pixel (W1)" , color = "blue")
plt.show() 