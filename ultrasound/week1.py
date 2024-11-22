# CFD HW4 - Combined Solution
# Chris Psenica
# 11/09/2024

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
from scipy.signal import hilbert, butter, filtfilt

#%---------- Problem Statement ----------
'''
For week1, this simulation needs to recreate the plots found in USFigs.pdf. This simulation is modeling a (water) immersed transducer which is sending waves to a block of aluminum. The distance from the transducer to the aluminum 
is 1cm (i.e. the wave from the transducer travels through 1cm of water before hitting the aluminum block). There is a crack in the aluminum block which is 1.5cm from the surface of the block. The total thickness of the block is
4cm. Upon reaching the end of the block the waves bounce back (of course, there is an echo not only from the bottom surface of the block but from the top surface as well as the crack inside of the block).
'''

#|==========================================================
#| Lab Parameters
#|==========================================================

#%---------- Given Parameters ----------
c_al = 6.2 * 1e3          #~ speed of wave in aluminum (m/s)
c_w  = 1.5 * 1e3          #~ speed of wave in water (m/s)
rho_al = 2700             #~ density of aluminum (kg/m^3)
rho_w = 1000              #~ density of aluminum (kg/m^3)
center_freq = 1e6         #~ transducer center frequency (Hz)
BW = 400e3                #~ -3dB transducer bandwidth (Hz)
water_dist = 10 * 1e-3    #~ initial water path distance (m)
crack_coeff = -0.2        #~ crack scatter coefficient
noise_amp = 22e3          #~ noise amplitude (Pa)
crack_depth = 1.5e-2      #~ depth of the flaw in the aluminum (m)
thickness_al = 4.5e-2     #~ total thickness of the block of aluminum (m)

#%---------- Frequency & Time Domain ----------
max_freq = 10e6                                           #~ upper bound on frequency range
max_time = 45e-6                                          #~ upper bound on time range
delta_t = 1 / max_freq                                    #~ time increment
delta_f = 1 / max_time                                    #~ frequency increment
n = int((delta_t * delta_f) ** -1)                        #~ number of sampling points
freq_base = np.arange(n , dtype = 'complex') * delta_f    #~ array of sampled frequencies 
time_base = np.arange(n , dtype = 'complex') * delta_t    #~ array of sampled time

#%---------- Plot Selection ----------
all_plots = True

#|==========================================================
#| Plot 1: Wave at source transducer in frequency domain
#|==========================================================
'''
This initial wave is at the transducer at time t = 0 seconds. we can construct this by using the Guassian function to an 
amplitude of 1 Pa*second.
'''

#%---------- Gaussian Function ----------
def G_amplitude1(f , c):

    B = ((c + 200e3) - c) / ((np.log(2 ** 0.5)) ** 0.5)    #~ B value for gaussian function
    Amp = np.zeros(int(len(f)) , dtype = "complex")        #~ initiate Amplitude matrix
    Amp[:] = 1 * np.exp(-((f[:] - c) / B) ** 2)            #~ populate Amplitude matrix

    return Amp

Amp1 = G_amplitude1(freq_base , 1e6)

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(freq_base * 1e-6 , Amp1 , color = "blue")
    plt.title("Wave at source transducer, frequency domain")
    plt.xlabel("Frequency MHz")
    plt.ylabel("Amplitude (Pascal Seconds)")
    plt.ylim((-0.01 , 1.01))
    plt.xlim((-0.1 , 10.1))
    plt.grid()
    plt.show()

#|==========================================================
#| Plot 2: Wave at source transducer in time domain
#|==========================================================
'''
This wave is created by taking the reverse fourier transform of the first frequency wave (Amp1) to take it from the 
frequency domain to the time domain.
'''

#%---------- Inverse FT ----------
def inv_ft2(Amp):

    length = int(len(Amp))
    amp = np.zeros(length , dtype = "complex")

    for i in range(length):
        sum = 0

        for s in range(length):
            sum += Amp[s] * np.exp(2j * np.pi * freq_base[s] * time_base[i]) * delta_f
            
        amp[i] = sum

    return amp

if all_plots:
    #%---------- Plot Results ----------
    amp2 = inv_ft2(Amp1)
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp2 , color = "blue")
    plt.title("Wave at source transducer, time domain")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-6e5 , 8e5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|==========================================================
#| Plot 3: Wave at water-solid boundary
#|==========================================================
'''
Here we must first convert the distance from the transducer to the top surface of the aluminum block into a time delay (t_delay).
Having this, we can create a phase shift (np.exp(-2j * np.pi * freq_base[:] * t_delay)) and apply this to the frequency domain (Amp1).
Taking the inverse FT of this will give the wave at the water/solid boundary.
'''

#%---------- Convert Spatial Distance & Inverse FT ----------
#^ get time delay from c_w
t_delay_water = water_dist / c_w

#^ calculate the new (phase shifted/ time delayed) frequency domain & inv. FT for time domain
Amp3 = Amp1 * np.exp(-2j * np.pi * freq_base[:] * t_delay_water)
amp3 = inv_ft2(Amp3)

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp3 , color = "blue")
    plt.title("Wave at water/solid boundary")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-6e5 , 6e5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|===============================================================================
#| Plot 4: Waveform detected by transducer, front surface reflection + scatter
#|===============================================================================
'''
For this plot we consider the wave being reflected from the top surface as well as the crack. We need the impedence for both domains (aluminum and water). From this we can calculate
what is transmitted from the water domain to the solid domain, what is reflected in the water domain by the solid domain and vice versa. The time delay must also be accounted
for as we are now travelling through a part of the aluminum block and back. When the transducer reads the incoming signal it goes through the Gaussian function again (this means
to multiply by Amp1 in the frequency domain) and then we can inverse FT to get the time domain plot.
'''

#%---------- Reflection & Transmission Coefficient ----------
#^ impedence for water and aluminum
Z_w = rho_w * c_w
Z_al = rho_al * c_al

#^ reflection and transmission coefficients
reflection1 = (Z_al - Z_w) / (Z_al + Z_w)    #~ reflection coefficient from aluminum to water (wave reflects off aluminum)
reflection2 = (Z_w - Z_al) / (Z_al + Z_w)    #~ reflection coefficient from water to aluminum (wave reflects off water)
transmission1 = (2 * Z_al) / (Z_al + Z_w)    #~ transmission coefficient from aluminum to water (wave disperses from water domain into solid domain) 
transmission2 = (2 * Z_w) / (Z_al + Z_w)     #~ transmission coefficient from water to aluminum (wave disperses from solid domain into water domain) 

#%---------- Inverse FT To Get Reflected Wave ----------
#^ calculate new delta_t from transmitted wave traveling through aluminum (up to crack) and then back
t_delay_aluminum = crack_depth / c_al
total_delay = (2 * t_delay_aluminum) + t_delay_water

#^ calculate the reflected and transmitted waves
Amp4_1 = Amp1 * Amp3 * reflection1 * np.exp(-2j * np.pi * freq_base[:] * t_delay_water)
Amp4_2 = Amp1 * Amp3 * transmission1 * transmission2 * crack_coeff * np.exp(-2j * np.pi * freq_base[:] * total_delay)

#^ inverse FT and sum the waves
amp4_1 = inv_ft2(Amp4_1) 
amp4_2 = inv_ft2(Amp4_2) 
amp4 = amp4_1 + amp4_2

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp4 , color = "blue")
    plt.title("Waveform detected by transducer, front surface reflection + scatter")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-4e5 , 4e5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|==============================================================================================
#| Plot 5: Waveform detected by transducer, front surface reflection + scatter + back surface
#|==============================================================================================
'''
This is the same as last plot but now we consider another wave reflection from the back surface. For this plot, we must considered what is transmitted from the
crack (not reflected back) and take into account the additional time delay from this. Assume that the wave is 100% reflected from the bottom surface.
'''

#%---------- Time Delay ----------
t_delay_aluminum_bottom = 2 * (thickness_al - crack_depth) / c_al
total_delay += t_delay_aluminum_bottom

#%---------- Inverse FT To Get Reflected Wave ----------
#^ calculate new wave in frequency domain
Amp4_3 = Amp1 * Amp3 * transmission1 * transmission2 * -(1 + crack_coeff) * np.exp(-2j * np.pi * freq_base[:] * total_delay)

#^ inverse FT to get new wave
amp4_3 = inv_ft2(Amp4_3) 
amp4 += amp4_3

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp4 , color = "blue")
    plt.title("Waveform detected by transducer, front surface reflection + scatter")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-4e5 , 4e5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|======================================================================================================
#| Plot 6: Waveform detected by transducer, front surface reflection + scatter + back surface + noise
#|======================================================================================================
'''
Add noise to Plot 5. For this we can use a Guassian White Noise generation. To do this, use np.random.normal(mean , std_dev, num_samples) * (noise_amp / std_dev).
Use mean = 0, std_dev = 1 for GWN. Add this to the time domain and plot.
'''

#%---------- Random Noise ----------
#^ generate noise
noise_array = np.random.normal(0 , 1 , int(len(Amp1))) * noise_amp

#^ add noise to simulation
amp4 += noise_array

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp4, color = "blue")
    plt.title("Waveform detected by transducer, front surface reflection + scatter + noise")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-4e5 , 4e5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|========================================================================
#| Plot 7: Hilbert transform of waveform detected by transducer + noise
#|========================================================================
'''
For this plot we do a Hilbert transform. We first must do a forward FT to get from the time domain into the frequency domain. Then, if there are any
negative frequencies we must zero them out. After doing this, inverse FT to get back into the time domain. We then use the real+imaginary parts
to get the magnitude. Plotting the magnitude gives the final result.
'''

#%---------- Forward FT ----------
def fd_ft2(amp):

    length = int(len(amp))
    Amp = np.zeros(length , dtype = "complex")

    for i in range(length):
        sum = 0

        for s in range(length):
            sum += amp[s] * np.exp(-2j * np.pi * freq_base[i] * time_base[s]) * delta_t
            
        Amp[i] = sum

    return Amp

Amp7 = fd_ft2(amp4) 
amp7 = inv_ft2(Amp7)
amp7[:] = ((amp7.real[:] ** 2.) + (amp7.imag[:] ** 2.)) ** 0.5

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp7 , color = "blue")
    plt.title("Hilbert transform of waveform detected by transducer + noise")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-15 , 5e5))
    plt.xlim((-0.5 , 50.5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()

#|=================================================================================
#| Plot 8: Hilbert transform of filtered waveform detected by transducer + noise
#|=================================================================================
'''
For this plot we do a Hilbert transform. We first must do a forward FT to get from the time domain into the frequency domain. Then, if there are any
negative frequencies we must zero them out. After doing this, inverse FT to get back into the time domain. We then use the real+imaginary parts
to get the magnitude. Plotting the magnitude gives the final result.
'''

#%---------- Filter ----------
b , a = butter(2 , [1.01e3, 2.15e3] , fs = delta_f , btype = 'band')
filtered_signal = filtfilt(b , a , amp4.real)
amp8 = np.abs(hilbert(filtered_signal))

if all_plots:
    #%---------- Plot Results ----------
    plt.figure(figsize = (14 , 10))
    plt.plot(time_base * 1e6 , amp8 , color = "blue")
    plt.title("Hilbert transform of filtered waveform detected by transducer + noise")
    plt.xlabel("Time (us)")
    plt.ylabel("Pressure (Pa)")
    plt.ylim((-15 , 2e5))
    plt.xlim((-0.5 , 50.5))
    plt.xlim((-0.5 , 50.5))
    plt.grid()
    plt.show()