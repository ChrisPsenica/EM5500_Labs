import numpy as np
import scipy.interpolate
import scipy as sp
from matplotlib import pyplot as pl

import pyopencl as cl

def load_materials(nlogE,*materials):
    """ Load material data from CSV saved from XCOM. 
    parameters:
      nlogE  -- number of steps in log Energy space for
                reconstruction
      materials:  additional parameters, each a 
                  (material file name, material density) tuple

      In place of up to one material file name you can use the
      special word "VACUUM" which will hotwire the corresponding 
      coefficients to 0

      Resulting material_data array will be indexed the same as 
      the materials list. 

    Assumes columns are: 
      Energy (MeV),  
      incoherent (Compton) scattering coefficient (cm^2/g)
      photoelectric absorption coefficient (cm^2/g)
    """
    cnt=0;

    log_reconstruction_list=[]
    density_list=[]
    
    for (filename,density) in materials:
        if filename=="VACUUM":
            assert(density==0.0)
            if cnt==0:
                # load arbitrary other file
                RawData = np.loadtxt(materials[1][0])
                pass
            else:
                # load arbitrary other file
                RawData = np.loadtxt(materials[0][0])
                pass
            pass
        else:
            RawData = np.loadtxt(filename)
            pass

        # Find locations where have discontinuities in the
        # energy distribution. 
        jumps = np.where((RawData[:-1,0]-RawData[1:,0])==0)[0]
        RawData_Index = np.arange(RawData.shape[0])

        curves = []  # list will have (start energy, end energy, compton_spline,photoe_spline) tuples
        for curveseg in range(jumps.shape[0]+1):
            if curveseg==0:
                start_logenergy=np.log10(RawData[0,0]*1e6)
                start_index=0
                pass
            else:
                start_logenergy=np.log10(RawData[jumps[curveseg-1]+1,0]*1e6)
                start_index=jumps[curveseg-1]+1
                pass
            if curveseg == jumps.shape[0]:
                end_logenergy=np.log10(RawData[RawData.shape[0]-1,0]*1e6) # last element
                end_index=RawData.shape[0]
                pass
            else:
                end_logenergy=np.log10(RawData[jumps[curveseg],0]*1e6)
                end_index=jumps[curveseg]+1
                pass
            
            
            curve_zone=(RawData_Index >= start_index) & (RawData_Index < end_index)
            
            # Perform spline fit for Compton and Photoelectric coefficients
            # Do curve-fit in log space ... (also convert from MeV->eV and cm^2/g -> m^2/kg: conversion : cm^2/g * (1m/100cm)^2 * (1000g / 1 kg)  = /10
            k=3
            if np.count_nonzero(curve_zone) <= k:
                # Cannot do a cubic fit if we only have a couple of
                # data points in this curve segment. Drop down to
                # quadratic or linear
                k=np.count_nonzero(curve_zone)-1
                pass
            
            compton_spl = sp.interpolate.splrep(np.log10(RawData[curve_zone,0]*1e6),np.log10(RawData[curve_zone,1]/10.0),k=k)
            photoe_spl = sp.interpolate.splrep(np.log10(RawData[curve_zone,0]*1e6),np.log10(RawData[curve_zone,2]/10.0),k=k)

            curves.append((start_logenergy,end_logenergy,compton_spl,photoe_spl))
            pass
        
        if cnt==0:
            # First time through, figure out logE, dlogE, etc.
            logE0=np.log10(RawData[0,0]*1e6)

            # Uniform grid of logE values
            logE=np.linspace(logE0,np.log10(RawData[-1,0]*1e6),nlogE)
            dlogE=logE[1]-logE[0]
            pass

        # Re-evaluate log10(compton) and log10(photoe)
        # at the uniform grid points based on the
        # spline fit of the various segments
        logcompton_recon = np.zeros(logE.shape,dtype='d')*np.NaN
        logphotoe_recon = np.zeros(logE.shape,dtype='d')*np.NaN
        for (start_logenergy,end_logenergy,compton_spl,photoe_spl) in curves:
            logErange = (logE >= start_logenergy) & (logE <= end_logenergy)
            logcompton_recon[logErange]=sp.interpolate.splev(logE[logErange],compton_spl,ext=2)
            logphotoe_recon[logErange]=sp.interpolate.splev(logE[logErange],photoe_spl,ext=2)
            pass
            
        log_reconstruction = (logcompton_recon,logphotoe_recon)

        if filename=="VACUUM":
            # hotwire output (absorption coefficients) to all zeros
            log_reconstruction[0][:]=0
            log_reconstruction[1][:]=0
            pass
        
        log_reconstruction_list.append(log_reconstruction)
        density_list.append(density)
        
        cnt+=1
        pass

    # Return material coefficients over a uniform grid
    # of logE values
    material_data = 10.0**np.array(log_reconstruction_list,dtype='f')
    # (material data values are not logarithms)
    
    material_rho = np.array(density_list,dtype='f')

    
    return (material_data,material_rho,cnt,logE0,dlogE,logE)



# Boundaries of voxels (all coordinates in meters)

# If your boundaries are symmetric and the number of
# positions is event, then you will get an element center
# right at the origin. 

x_bnd = np.linspace(-.1,.1,20).astype('f')
y_bnd = np.linspace(-.15,.15,30).astype('f')

dx=x_bnd[1]-x_bnd[0]
dy=y_bnd[1]-y_bnd[0]

z_bnd=np.linspace(0,.14,20).astype('f')
dz=z_bnd[1]-z_bnd[0]

# z coordinate of source
source_z=0.0


# Detector z
detector_z=.14 # (CODE ASSUMES THAT THIS IS AT OR BEYOND THE LAST z_bnd)


# locations of centers of voxels
x = (x_bnd[0:-1]+x_bnd[1:])/2.0
y = (y_bnd[0:-1]+y_bnd[1:])/2.0
z = (z_bnd[0:-1]+z_bnd[1:])/2.0

nx=x.shape[0]
ny=y.shape[0]
nz=z.shape[0]

# In MATLAB you would use NDGRID instead of MESHGRID....
# MESHGRID in MATLAB or MESHGRID with indexing="xy"
# in Python flips axes in weird ways.
(x_2d,y_2d) = np.meshgrid(x,y,indexing="ij")  

(x_3d,y_3d,z_3d) = np.meshgrid(x,y,z,indexing="ij")

material_volumetric=np.zeros((nx,ny,nz),dtype=np.uint32)
# Insert material selections into material_volumetric according to material indices below

## Column of aluminum down the middle:
#material_volumetric[(nx-1)/2,(ny-1)/2,:]=1

# Box of aluminum
# Note that you need extra parentheses beyond what you
# would normally need in an 'if' statement!
# (in MATLAB you would use the find() function with a
# similar criterion) 
material_volumetric[(x_3d > -.05) & (x_3d < .05) & (y_3d > -.05) & (y_3d < .05) & (z_3d < .05)]=1

nlogE=400


# Here is where materials are specified by
# data file and density.
#
# You will define your material cube by indexes
# of these materials.
#
# They are specified by filename (or the special word VACUUM)
# and density in kg/m^3
#
# The file should have 3 columns: Energy in MeV,
# Compton coefficient (cm^2/g) and photoelectric
# coefficient (cm^2/g)

(material_data,
 material_rho,
 n_materials,
 logE0,
 dlogE,
 logE) = load_materials(nlogE,
                        ("VACUUM",0.0), # material #0
                        ("Aluminum.dat",2700.0), # matl #1
                        ("Iron.dat",7900.0), # matl #2
                        ("SiO2.dat",2600.0)) # matl #3




n_materials=material_data.shape[0]
assert(n_materials==material_rho.shape[0])
assert(material_data.shape[2]==nlogE)
assert(material_data.shape[1]==2) # Compton and Photoe



# Here is where we specify the initial positions and
# directions of the photons.
#
# The number of initial positions is arbitrary.
# If a given position is included multiple times
# you will get proportionally more photons eminatting
# from that position 


## These next three lines give an array of initial
## positions that line up with the voxel grid
## and are located at z=source_z
photonpos_x=x_2d.copy()
photonpos_y=y_2d.copy()
photonpos_z = (source_z*np.ones(x_2d.shape,dtype='f'))

## These next three lines give a concentrated columnar source
## at (x=0,y=0,z=xource_z...
## the 100,100 starts 10,000 beams together
#photonpos_x=np.zeros((100,100))
#photonpos_y=np.zeros((100,100))
#photonpos_z=source_z*np.ones((100,100))

# Assemble above position arrays into a single array
# where the rightmost axis represents x, y, or z
photonpos=np.zeros((photonpos_x.shape[0],photonpos_x.shape[1],3),dtype='f')
# photonpos is now nx * ny * 3
photonpos[:,:,0]=photonpos_x
photonpos[:,:,1]=photonpos_y
photonpos[:,:,2]=photonpos_z


# Here is where we specify the initial directions of
# the photons.
# each entry in photonvec should be a unit vector

## These next few lines initialize all photons to 
## be parallel rays in the +z direction.
photonvec=np.zeros((photonpos_x.shape[0],photonpos_x.shape[1],3),dtype='f')
photonvec[:,:,0] = 0
photonvec[:,:,1] = 0
photonvec[:,:,2] = 1.0
# photonvec is now nx * ny * 3

# Here is where we specify how many photons
# to start at each of the positions specified
# above
#num_photons_per_postion=100000 
num_photons_per_position=10000

# Number of photons per position to include
# in each kernel execution.
# If graphics stops while the calculation
# is running (due to the compute hogging
# the GPU), or if the output seems incomplete,
# (due to the GPU timing out) you can
# try reducing this.
photons_per_position_per_kernel_run=100

# Here is where we specify the energy of
# the incoming photons. It is stored as
# the log10 of the energy in eV

## These next few lines select a single
## energy for all incoming photons
photon_energy = 50e3 # energy in eV
photon_logE = np.log10(photon_energy*np.ones(photonpos.shape[:2],dtype='f')).astype('f') 



# The detector is assumed to have the same
# x,y shape, size, and position as the voxel grid. 
detector_photons=np.zeros(x_2d.shape,dtype=np.int32)


# pyopencl initialization
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

# Create buffers for transferring data to/from GPU
material_data_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=material_data)
material_rho_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=material_rho)

material_volumetric_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=material_volumetric)


photonpos_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=photonpos)
photonvec_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=photonvec)
photon_logE_buf = cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=photon_logE)


detector_photons_buf = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=detector_photons)

# Load in and compile the GPU code
c_code = open('XRayGPU.c',"r").read()
prg = cl.Program(ctx,c_code);
prg.build()

absorp_kernel=prg.absorp # absorp is name of C function

# These are the data types of the scalar parameters to
# the compute kernel. They MUST be kept
# consistent with the kernel function parameters
absorp_kernel.set_scalar_arg_dtypes([
    np.float32,np.float32,np.uint32,
    np.float32,np.float32,np.uint32,
    np.float32,np.float32,np.uint32,
    np.float32,
    np.float32,np.float32,np.uint32,
    None,None,
    np.uint32,
    None,
    np.uint32,
    None,None,
    None,
    np.uint32,
    None,
    np.uint32])

# Iterate over all of the photons we are
# trying to generate from each position

while num_photons_per_position > 0:

    # Compute number of photons per position for this iteration
    use_photons=min(photons_per_position_per_kernel_run,num_photons_per_position)

    # compute_dimensions is how many parallel threads we will have
    compute_dimensions = photonpos.shape[:2]

    # Run the compute kernel
    res=absorp_kernel(queue,compute_dimensions,None,
                      # Parameters to C function start here
                      x_bnd[0],dx,x_bnd.shape[0],
                      y_bnd[0],dy,y_bnd.shape[0],
                      z_bnd[0],dz,z_bnd.shape[0],
                      detector_z,
                      logE0,dlogE,nlogE,
                      material_data_buf, material_rho_buf,
                      n_materials,
                      material_volumetric_buf,
                      compute_dimensions[1],
                      photonpos_buf,photonvec_buf,
                      photon_logE_buf,
                      use_photons,
                      detector_photons_buf,
                      int(np.random.rand()*(2**32-1.0)))
    num_photons_per_position -= use_photons
    pass

# Copy result back from GPU
cl.enqueue_copy(queue,detector_photons,detector_photons_buf,wait_for=(res,),is_blocking=True)

# Generate plot
pl.figure(1)
pl.clf()
pl.imshow(detector_photons.T,origin='lower',extent=(x_bnd[0]*1e3,x_bnd[-1]*1e3,y_bnd[0]*1e3,y_bnd[-1]*1e3),vmin=0,vmax=np.max(detector_photons))
pl.colorbar()
pl.title("Photon count in each detector pixel")
pl.xlabel('X position (mm)')
pl.ylabel('Y position (mm)')


pl.show()
