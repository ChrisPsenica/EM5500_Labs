import numpy as np
import matplotlib.pyplot as plt
class Material:
    def __init__(self, name, density, mu_pa, mu_com):
        self.name = name
        self.density = density  # g/cm^3
        self.mu_pa = mu_pa  # Photoelectric absorption coefficient
        self.mu_com = mu_com  # Compton scattering coefficient
class Specimen:
    def __init__(self, size, background_material, contrast_material, contrast_shape):
        self.size = size
        self.background_material = background_material
        self.contrast_material = contrast_material
        self.contrast_shape = contrast_shape
        self.volume = np.zeros(size)
        self._create_contrast()
    def _create_contrast(self):
        x, y, z = np.indices(self.size)
        if self.contrast_shape == 'sphere':
            center = np.array(self.size) // 2
            radius = min(self.size) // 4
            sphere = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
            self.volume[sphere] = 1
class Detector:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance
        self.image = np.zeros(size)
def klein_nishina(energy, cos_theta):
    # Energy in keV
    alpha = energy / 511  # 511 keV is the rest mass energy of an electron
    r_e = 2.818e-15  # classical electron radius in meters
    term1 = (1 / (1 + alpha * (1 - cos_theta)))**2
    term2 = (1 + cos_theta**2) / 2
    term3 = 1 + (alpha**2 * (1 - cos_theta)**2) / ((1 + cos_theta**2) * (1 + alpha * (1 - cos_theta)))
    return r_e**2 * term1 * term2 * term3
def calculate_photon_interactions(volume, mu_pa_bg, mu_com_bg, mu_pa_ct, mu_com_ct,
                                  detector_size, detector_distance, I0, energy):
    specimen_size = volume.shape
    photoelectric_absorbed = np.zeros_like(volume)
    compton_scattered = np.zeros_like(volume)
    transmitted = np.full_like(volume, I0, dtype=np.float64)
    for z in range(specimen_size[2]):
        for y in range(specimen_size[1]):
            for x in range(specimen_size[0]):
                if volume[x, y, z] == 1:
                    mu_pa, mu_com = mu_pa_ct, mu_com_ct
                else:
                    mu_pa, mu_com = mu_pa_bg, mu_com_bg
                mu_total = mu_pa + mu_com
                dz = 1  # Assuming each voxel is 1 unit thick
                absorbed = transmitted[x, y, z] * (1 - np.exp(-mu_total * dz))
                photoelectric_absorbed[x, y, z] = absorbed * (mu_pa / mu_total)
                compton_scattered[x, y, z] = absorbed * (mu_com / mu_total)
                if z < specimen_size[2] - 1:
                    transmitted[x, y, z+1] = transmitted[x, y, z] * np.exp(-mu_total * dz)
    detector_image = transmitted[:, :, -1]
    # Calculate Compton scattering to detector
    compton_to_detector = np.zeros(detector_size, dtype=np.float64)
    for z in range(specimen_size[2]):
        for y in range(specimen_size[1]):
            for x in range(specimen_size[0]):
                if compton_scattered[x, y, z] > 0:
                    for dy in range(detector_size[0]):
                        for dx in range(detector_size[1]):
                            dx_rel = dx - x
                            dy_rel = dy - y
                            dz_rel = detector_distance
                            r = np.sqrt(dx_rel**2 + dy_rel**2 + dz_rel**2)
                            cos_theta = dz_rel / r
                            solid_angle = 1 / (4 * np.pi * r**2)
                            compton_to_detector[dy, dx] += compton_scattered[x, y, z] * klein_nishina(energy, cos_theta) * solid_angle
    return photoelectric_absorbed, compton_scattered, detector_image, compton_to_detector
# Load data and calculate coefficients:
iodine_dat = np.loadtxt('iodine.dat')
tungsten_dat = np.loadtxt('tung.dat')
# Define arrays:
eng = iodine_dat[:, 0]
al_com_io = iodine_dat[:, 1]
al_pa_io = iodine_dat[:, 2]
al_com_tg = tungsten_dat[:, 1]
al_pa_tg = tungsten_dat[:, 2]
# Evaluate mu for contrast and background material:
mu_pa_io = al_pa_io * 4.93  # Using the density of Iodine
mu_com_io = al_com_io * 4.93
mu_pa_tg = al_pa_tg * 19.28  # Using the density of Tungsten
mu_com_tg = al_com_tg * 19.28
# Create materials
background = Material("Tungsten", 19.28, mu_pa_tg, mu_com_tg)
contrast = Material("Iodine", 4.93, mu_pa_io, mu_com_io)
# Create specimen (reduced size)
specimen_size = (20, 20, 20)  # Reduced size
specimen = Specimen(specimen_size, background, contrast, 'sphere')
# Create detector (reduced size)
detector_size = (20, 20)  # Reduced size
detector_distance = 15
detector = Detector(detector_size, detector_distance)
# Set initial intensity and energy
I0 = 1e8  # Reduced number of photons/sec
energy_index = 55
print("Running calculation...")
photoelectric_absorbed, compton_scattered, detector_image, compton_to_detector = calculate_photon_interactions(
    specimen.volume,
    background.mu_pa[energy_index],
    background.mu_com[energy_index],
    contrast.mu_pa[energy_index],
    contrast.mu_com[energy_index],
    detector.size,
    detector.distance,
    I0,
    eng[energy_index]
)
# Visualize results
plt.figure(0)
plt.imshow(detector_image, cmap='hot')
plt.title("Detector Contour")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.figure(1)
plt.imshow(np.sum(photoelectric_absorbed, axis=2), cmap='hot')
plt.title("Total Photoelectric Absorption")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.figure(2)
plt.imshow(compton_to_detector, cmap='hot')
plt.title("Compton Scattering to Detector")
plt.colorbar()
plt.tight_layout()
plt.show()
# Calculate the total number of incident photons per second
total_incident_photons_per_sec = I0 * specimen.size[0] * specimen.size[1]
# Calculate the total number of photoelectric absorbed, Compton scattered, and transmitted photons per second
total_photoelectric_absorbed_per_sec = np.sum(photoelectric_absorbed)
total_compton_scattered_per_sec = np.sum(compton_scattered)
total_transmitted_per_sec = np.sum(detector_image)
total_compton_to_detector_per_sec = np.sum(compton_to_detector)
# Print additional information
print(f"Specimen size: {specimen.size}")
print(f"Detector size: {detector.size}")
print(f"Detector distance: {detector.distance}")
print(f"Background material: {specimen.background_material.name}, density: {specimen.background_material.density} g/cm^3")
print(f"Contrast material: {specimen.contrast_material.name}, density: {specimen.contrast_material.density} g/cm^3")
print(f"Energy level: {eng[energy_index]} keV")
print(f"Incident intensity: {I0:.2e} photons/sec/pixel")
print(f"Total incident photons per second: {total_incident_photons_per_sec:.2e} photons/sec")
print(f"Total photoelectric absorbed photons per second: {total_photoelectric_absorbed_per_sec:.2e} photons/sec")
print(f"Total Compton scattered photons per second: {total_compton_scattered_per_sec:.2e} photons/sec")
print(f"Total transmitted photons per second: {total_transmitted_per_sec:.2e} photons/sec")
print(f"Total Compton scattered photons reaching detector per second: {total_compton_to_detector_per_sec:.2e} photons/sec")
# Calculate and print percentages
print(f"\nPercentages:")
print(f"Photoelectric absorption: {100 * total_photoelectric_absorbed_per_sec / total_incident_photons_per_sec:.2f}%")
print(f"Compton scattering: {100 * total_compton_scattered_per_sec / total_incident_photons_per_sec:.2f}%")
print(f"Transmission: {100 * total_transmitted_per_sec / total_incident_photons_per_sec:.2f}%")
print(f"Compton scattering reaching detector: {100 * total_compton_to_detector_per_sec / total_incident_photons_per_sec:.2f}%")
# Check conservation of photons
total_accounted = total_photoelectric_absorbed_per_sec + total_compton_scattered_per_sec + total_transmitted_per_sec
print(f"\nConservation check:")
print(f"Total accounted photons per second: {total_accounted:.2e}")
print(f"Difference from incident: {total_accounted - total_incident_photons_per_sec:.2e}")
print(f"Percent difference: {100 * (total_accounted - total_incident_photons_per_sec) / total_incident_photons_per_sec:.4f}%")
# Klein-Nishina specific information
print(f"\nKlein-Nishina specific information:")
print(f"Total Compton scattered photons: {total_compton_scattered_per_sec:.2e}")
print(f"Compton scattered photons reaching detector: {total_compton_to_detector_per_sec:.2e}")
print(f"Fraction of Compton scattered photons reaching detector: {total_compton_to_detector_per_sec / total_compton_scattered_per_sec:.4f}")
# Energy dependence
print(f"\nEnergy dependence:")
print(f"Energy: {eng[energy_index]} keV")
print(f"Klein-Nishina cross-section at 90 degrees: {klein_nishina(eng[energy_index], 0):.2e} m^2/sr")
# Angular distribution
angles = np.linspace(0, np.pi, 181)
cross_sections = [klein_nishina(eng[energy_index], np.cos(angle)) for angle in angles]
max_angle = angles[np.argmax(cross_sections)] * 180 / np.pi
print(f"\n Angular distribution:")
print(f"Angle of maximum scattering: {max_angle:.2f} degrees")
plt.figure(figsize=(10, 6))
plt.plot(angles * 180 / np.pi, cross_sections)
plt.title(f"Klein-Nishina Cross-Section vs. Scattering Angle at {eng[energy_index]} keV")
plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Cross-Section (m^2/sr)")
plt.yscale('log')
plt.grid(True)
plt.show()