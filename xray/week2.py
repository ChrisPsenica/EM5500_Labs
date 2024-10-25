# XRay Lab
# Chris Psenica

#---------- imports ----------
import numpy as np
import matplotlib.pyplot as plt
import time
from mat_data import tungsten_p , copper_p , tungsten_c , copper_c , air_c , air_p , silicon_c , silicon_p , cdte_p , cdte_c

#---------- Input Parameters ----------
print()
print("PHOTON SIMULATION")
photon_energy = float(input("Enter the initial photon energy (values can range from 1e-3 - 10): "))    # energy of the photon in MeV
detector_material = str(input("Enter the detector material (silicon or cdte): "))                    # pick which detector material to simulate. Options are silicon and cdte (cadmium telluride)
z_detect = float(input("Enter the detector distance from voxel grid: "))                               # z location of the detector
print("Beginning Sumulation" , end = '' , flush = True)
for i in range(5):
    print('.' , end = '' , flush = True)
    time.sleep(0.3)  # Wait 1 second between each period
print("Start!")
print()
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
grid_points = 51        # resolution (pixel density)

lower_bnd = -5
upper_bnd = 5

nx = 11   # number of points in x direction for voxel grid
ny = 13   # number of points in y direction for voxel grid
nz = 9   # number of points in z direction for voxel grid

mat_array = np.zeros((nx , ny , nz) , dtype = np.uint32)   # voxel grid (material array)
x = np.linspace(lower_bnd , upper_bnd , nx)                # x range
y = np.linspace(lower_bnd , upper_bnd , ny)                # y range
z = np.linspace(lower_bnd, upper_bnd , nz)                 # z range

(X , Y , Z) = np.meshgrid(x , y , z , indexing = "ij")    # initiate meshgrid for 3D x , y , z with center element at (x , y , z) = (0 , 0 , 0)
r_outer = 3                                               # outer radius of tungsten sphere is 3
r_inner = 3                                               # outer radius of copper sphere is 1

mat_array[(X ** 2. + Y ** 2. + Z ** 2. < r_outer ** 2.) & (X > 0)] = 2      # makes a tungsten half sphere at the center of voxel grid
mat_array[(X ** 2. + Y ** 2. + Z ** 2. < r_inner ** 2.) & (X < 0)] = 3      # makes a copper half sphere at the center of voxel grid

delta_x = (upper_bnd - lower_bnd) / nx    # delta_x
delta_y = (upper_bnd - lower_bnd) / ny    # delta_y

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
    alpha = photon_energy / (512e-3)
    mew = table[index , 1] * density

    return mew , alpha

#---------- Calculate mew ----------                                                    # mew for the vacuum is zero
mew_copper_c , alpha_copper_c = get_mew(photon_energy , copper_c , density[2])          # mew -> coefficient of absorption for copper for compton scattering
mew_copper_p , alpha_copper_p = get_mew(photon_energy , copper_p , density[2])          # mew -> coefficient of absorption for copper for photoelectric absorption
mew_tungsten_c , alpha_tungsten_c = get_mew(photon_energy , tungsten_c , density[1])    # mew -> coefficient of absorption for tungsten for compton scattering
mew_tungsten_p , alpha_tungsten_p = get_mew(photon_energy , tungsten_p , density[1])    # mew -> coefficient of absorption for tungsten for photoelectric absorption
mew_air_c , alpha_air_c = get_mew(photon_energy , air_c , density[3])                   # mew -> coefficient of absorption for air for compton scattering
mew_air_p , alpha_air_p = get_mew(photon_energy , air_p , density[3])                   # mew -> coefficient of absorption for air for photoelectric absorption

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
    I        = Total intensity
    not_comp = Photons which are not compton scattered
    Ic       = Intensity from compton scattering
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

    I = I_i * np.exp(-(mew_p + mew_c) * dz)  
    Ic = I_i * (1 - np.exp(-mew_c * dz))
    not_comp = I_i * np.exp(-mew_c * dz)

    return not_comp , I , Ic

#---------- KN Formula ----------
def get_r_theta(k , s , pixel_val):
    '''
    inputs:
    -------
    k         = z index on voxel grid
    s         = y index on Icp matrix (for calculating scattered photons to each pixel)
    pixel_val = Identification number of what material the pixel represents

    outputs:
    --------
    r     = Distance from voxel to pixel
    theta = Inclination angle from voxel to pixel
    alpha = Alpha value from NIST XCOM (this value can be found in mat_data.py)
    '''

    if pixel_val == 0: # vacuum
        alpha = 0

    elif pixel_val == 1: # air
        alpha = alpha_air_c

    elif pixel_val == 2: # tungsten
        alpha = alpha_tungsten_c

    elif pixel_val == 3: # copper
        alpha = alpha_copper_c

    r1 = (upper_bnd - lower_bnd) + z_detect - (dz * k)
    r2 = abs(y[j] - y[s])

    r = ((r1 ** 2) + (r2 ** 2)) ** 0.5
    theta = np.arccos(r1 / r)

    return r , theta , alpha

def IcpIc(alpha , theta , r):
    '''
    inputs:
    -------
    r     = Distance from voxel to pixel
    theta = Inclination angle from voxel to pixel
    alpha = Alpha value from NIST XCOM (this value can be found in mat_data.py)

    outputs:
    --------
    IcpIc = The ratio Icp/Ic
    '''

    KN = ((1 / (1 + (alpha * (1 - np.cos(theta))))) ** 2) * ((1 + (np.cos(theta) ** 2.)) / 2.) * (1 + (((alpha * (1 - np.cos(theta))) ** 2) / ((1 + ((np.cos(theta)) ** 2)) * (1 + (alpha * (1 - np.cos(theta)))))))
    delta_ohm = (delta_x * delta_y * np.cos(theta)) / (r **2)
    Ic_analytical = ((1 + alpha) / (alpha ** 2)) * (((2 * (1 + alpha)) / (1 + (2 * alpha))) - ((1 / alpha) * np.log10(1 + (2 * alpha)))) + ((1 / (2 * alpha)) * np.log10(1 + (2 * alpha))) - ((1 + (3 * alpha)) / ((1 + (2 * alpha)) ** 2))

    return (KN * delta_ohm) / Ic_analytical

#---------- Initiate Matrices ----------
solution_array = np.zeros((nx , ny))    # array to store solution (photons per pixel)
I_array = np.zeros((nx , ny))           # array to store new I value (holds renewed I_0 values to use for next layer in voxel grid)
Icp = np.zeros((nx , ny))               # array to store Icp/Ic values calculated per voxel
Ic_array = np.zeros((nx , ny))          # array to store Ic values (solution = Icp/Ic * Ic)
I_array[:] = I_0                        # populate I_array with I_0
z = 0                                   # parameter for while loop
iteration = 0                           # initial iteration value (used to print out iteration number)

#---------- Iterate Over Each Layer Of Voxel Grid ----------
for k in range(nz):         # iterate in z-direction
    for i in range(nx):   
        for j in range(ny): # iterate over face of voxel grid (x-y plane)

            print("Iteration:" , iteration)
            for c in range(nx):
                for s in range(ny):
                    r , theta , alpha = get_r_theta(k , s , int(mat_array[i , j , k]))
                    if int(mat_array[i , j , k]) != 0:
                        Icp[c , s] = IcpIc(alpha , theta , r)
            iteration += 1
            solution_array[i , j] , I_array[i , j] , Ic_array[i , j] = photons(I_array[i , j] , int(mat_array[i , j , k]))
            solution_array += Icp * Ic_array

z_air = 0
while z_air <= z_detect:    # while iterating through air (between the voxel grid and detector)
    for i in range(nx):    
        for j in range(ny): # iterate over face of voxel grid (x-y plane)

            print("Iteration:" , iteration)
            for c in range(nx):
                for s in range(ny):
                    r , theta , alpha = get_r_theta(k , s , 1)
                    if int(mat_array[i , j , k]) != 0:
                        Icp[c , s] = IcpIc(alpha , theta , r)
            iteration += 1
            solution_array[i , j] , I_array[i , j] , Ic_array[i , j] = photons(I_array[i , j] , 1)
            solution_array += Icp * Ic_array

    z_air += dz

#---------- Plot Photons ----------
c = plt.imshow(solution_array.T , cmap = 'rainbow' , vmin = 0 , vmax = 1.2 * I_0 , extent = [x.min() , x.max() , y.min() , y.max()] , interpolation = 'nearest' , origin = 'lower') 
plt.colorbar(c) 
plt.xlabel("X Location On Sensor")
plt.ylabel("Y Location On Sensor")
plt.title("Sensor Readout Of The Number Of Photons Per Pixel (W2)" , color = "blue")
plt.show() 

#---------- Description: Photons Absorbed Or Scattered On The Detector ----------
'''
Here estimate the number of photons which actually interact with the detector. Choose a detector material (silicone or cadmium telluride). 
A photon interacts with the detector pixel iff it is absorbed photoelectrically or is Compton scattered. The thickness of the detector will be 0.5 mmm. 
Since our measurements all assume cm (e.g. global distance unit is cm), then this value is converted to cm.

For this problem, because why not, both detector materials will be modeled. 
'''
#---------- Densities ----------
dz_detector = 0.5 * 1e-1         # thickness of the detector in cm

# 0 -> silicon   | density = 2.33   g/cm^3 
# 1 -> CdTe      | density = 5.85   g/cm^3
density_detector = [2.33 , 5.85]    # array of density values [silicon , CdTe] in g/cm^3

#---------- Calculate mew ----------
mew_silicon_c , _ = get_mew(photon_energy , silicon_c , density_detector[0])    #? mew -> coefficient of absorption for copper for compton scattering
mew_silicon_p , _ = get_mew(photon_energy , silicon_p , density_detector[0])    #? mew -> coefficient of absorption for copper for photoelectric absorption
mew_cdte_c , _ = get_mew(photon_energy , cdte_c , density_detector[1])          #? mew -> coefficient of absorption for copper for compton scattering
mew_cdte_p , _ = get_mew(photon_energy , cdte_p , density_detector[1])          #? mew -> coefficient of absorption for copper for photoelectric absorption

if detector_material == "silicon":
    mew_c = mew_silicon_c
    mew_p = mew_silicon_p

elif detector_material == "cdte":
    mew_c = mew_cdte_c
    mew_p = mew_cdte_p
else:
    raise ValueError("Invalid material selected for the detector. Please choose either Silicon (silicon) or Cadmium Telluride (cdte)")

def detector_interactions(I_i):

    is_photo_absorbed = I_i * (1 - np.exp(-mew_p * dz_detector))
    is_comp_scattered = I_i * (1 - np.exp(-mew_c * dz_detector))

    return is_photo_absorbed + is_comp_scattered 

#---------- Iterate Over Detector Face ----------
for i in range(nx):        # iterate over x direction
    for j in range(ny):    # iterate over y direction

        solution_array[i , j] = detector_interactions(solution_array[i , j])

#---------- Plot Detector Interactions ----------
vmax = np.amax(solution_array) * 1.2
c = plt.imshow(solution_array.T , cmap = 'rainbow' , vmin = 0 , vmax = vmax , extent = [x.min() , x.max() , y.min() , y.max()] , interpolation = 'nearest' , origin = 'lower') 
plt.colorbar(c) 
plt.xlabel("X Location On Sensor")
plt.ylabel("Y Location On Sensor")
plt.title("Sensor Readout Of The Number Of Photon Interactions Per Pixel (W2)" , color = "blue")
plt.show() 