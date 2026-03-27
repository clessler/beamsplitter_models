import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

c = 2.9979e8  # speed of light in m/s

# list of potential beamsplitter materials; indices are (n, eps), with n = Re(N) and eps = real part of complex permittivity
# all code done using the complex refractive index, N, as calculated from n and eps

material_indices = [(1.5141,2.2923),(1.611, 2.595), (1.954, 3.82), (2.11, 4.45), (1.5246, 2.3244), (2.37, 5.62), (1.73, 2.993) \
                    , (1.48, 2.19), (1.5185, 2.3058), (1.83, 3.35), (1.608, 2.585), (1.6, 2.56), (1.45, 2.1), (1.8738, 3.511), \
                        (1.572, 2.47), (3.4464, 11.878), (1.43, 2.04), (1.50, 2.25)]
material_names = ['LDPE', 'Acrylic 31', 'Spectrosil', 'Pyrex', 'HDPE', 'Macor', 'Nylon', 'Paraffin', 'PE', 'PETP', 'PMMA', 'PS', \
                  'PTFE(Teflon)', 'Fused Quartz (Herasil)', 'Rexolite', 'Silicon', 'TPX (Sheet)', 'PP']

material_dict = dict(zip(material_names, material_indices))


# function to print the available material names
def available_materials():
    for name in material_names:
        print(name)

# s-pol at oblique incidence; for a slab in air
def oblique_slab_reflection_s(n, eps, h, nu, theta_1):
    k = 2*np.pi*nu/c # wavenumber in vacuum
    N = n + 1j*np.emath.sqrt(n**2 -eps) # complex refractive index
    k1z = k*np.cos(theta_1)
    theta_2 = np.arcsin(np.sin(theta_1)/n) # Snell's law
    theta_3 = theta_1 # air on both sides
    k2z = k*N*np.cos(theta_2)
    k3z = k*np.cos(theta_3)

    M21 = 0.5*(1 - (k1z/k3z))*np.cos(k2z*h) - 0.5j*((k2z/k3z)-(k1z/k2z))*np.sin(k2z*h)
    M22 = 0.5*(1 + (k1z/k3z))*np.cos(k2z*h) - 0.5j*((k2z/k3z)+(k1z/k2z))*np.sin(k2z*h)
    r_s = -M21/M22
    R_s = np.abs(r_s)**2
    return R_s

# p-pol at oblique incidence; for a slab in air
def oblique_slab_reflection_p(n, eps, h, nu, theta_1):
    k = 2*np.pi*nu/c # wavenumber in vacuum
    N = n + 1j*np.emath.sqrt(n**2 -eps) # complex refractive index
    eps_2 = eps -1j*2*n*np.emath.sqrt(n**2 -eps) # complex permittivity
    eps_1 = 1
    eps_3 = 1 # air on both sides
    k1z = k*np.cos(theta_1)
    theta_2 = np.arcsin(np.sin(theta_1)/n) # Snell's law
    theta_3 = theta_1 # air on both sides
    k2z = k*N*np.cos(theta_2)
    k3z = k*np.cos(theta_3)

    M21 = 0.5*(1 - (eps_3/eps_1)*(k1z/k3z))*np.cos(k2z*h) - 0.5j*((eps_3/eps_2)*(k2z/k3z)-(eps_2/eps_1)*(k1z/k2z))*np.sin(k2z*h)
    M22 = 0.5*(1 + (eps_3/eps_1)*(k1z/k3z))*np.cos(k2z*h) - 0.5j*((eps_3/eps_2)*(k2z/k3z)+(eps_2/eps_1)*(k1z/k2z))*np.sin(k2z*h)
    r_p = -M21/M22
    R_p = np.abs(r_p)**2
    return R_p

def oblique_slab_transmission_s(n, eps, h, nu, theta_1):
    """
    s-polarization power transmittance for a dielectric slab in air.
    Uses transmission-line matrix method.
    
    Parameters:
        n: real part of refractive index
        eps: real part of permittivity (used to compute imaginary part of N)
        h: slab thickness [m]
        nu: frequency [Hz]
        theta_1: angle of incidence [rad]
    
    Returns:
        T_s: power transmittance for s-polarization
    """
    k = 2 * np.pi * nu / c  # wavenumber in vacuum
    N = n + 1j * np.emath.sqrt(n**2 - eps)  # complex refractive index
    
    k1z = k * np.cos(theta_1)  # z-component of wavevector in air (incident)
    theta_2 = np.arcsin(np.sin(theta_1) / n)  # Snell's law (real part)
    k2z = k * N * np.cos(theta_2)  # z-component in slab
    k3z = k * np.cos(theta_1)  # z-component in air (transmitted), same as k1z
    
    # M22 element of transfer matrix for s-polarization
    M22 = 0.5 * (1 + (k1z / k3z)) * np.cos(k2z * h) - 0.5j * ((k2z / k3z) + (k1z / k2z)) * np.sin(k2z * h)
    
    # Transmission coefficient: t = 1/M22 (for symmetric slab, input/output impedances equal)
    t_s = 1.0 / M22
    T_s = np.abs(t_s)**2
    
    return T_s


def oblique_slab_transmission_p(n, eps, h, nu, theta_1):
    """
    p-polarization power transmittance for a dielectric slab in air.
    Uses transmission-line matrix method.
    
    Parameters:
        n: real part of refractive index
        eps: real part of permittivity (used to compute imaginary part of N)
        h: slab thickness [m]
        nu: frequency [Hz]
        theta_1: angle of incidence [rad]
    
    Returns:
        T_p: power transmittance for p-polarization
    """
    k = 2 * np.pi * nu / c  # wavenumber in vacuum
    N = n + 1j * np.emath.sqrt(n**2 - eps)  # complex refractive index
    eps_2 = eps - 1j * 2 * n * np.emath.sqrt(n**2 - eps)  # complex permittivity of slab
    eps_1 = 1  # air
    eps_3 = 1  # air
    
    k1z = k * np.cos(theta_1)
    theta_2 = np.arcsin(np.sin(theta_1) / n)  # Snell's law
    k2z = k * N * np.cos(theta_2)
    k3z = k * np.cos(theta_1)
    
    # M22 element of transfer matrix for p-polarization
    M22 = 0.5 * (1 + (eps_3 / eps_1) * (k1z / k3z)) * np.cos(k2z * h) \
        - 0.5j * ((eps_3 / eps_2) * (k2z / k3z) + (eps_2 / eps_1) * (k1z / k2z)) * np.sin(k2z * h)
    
    # Transmission coefficient
    t_p = 1.0 / M22
    T_p = np.abs(t_p)**2
    
    return T_p

def beamsplitter_reflectance(nus, h, theta, material=None, n=None, eps=None):
    '''
    n = real part of refractive index
    eps = real part of complex permittivity
    h = thickness of slab in m
    nus = list of frequencies in Hz
    theta = angle of incidence in degrees

    Returns:
    R_p = reflectance (fraction of incident power) for p-polarized light
    R_s = reflectance for s-polarized light
    R_avg = reflectance for unpolarized light
    '''
    if material is not None:
        n, eps = material_dict[material]
    elif n is None or eps is None:
        raise ValueError("You must provide both n and eps when material is not specified.")
    
    theta_rad = np.deg2rad(theta)
    R_p = oblique_slab_reflection_p(n, eps, h, nus, theta_rad)
    R_s = oblique_slab_reflection_s(n, eps, h, nus, theta_rad)
    R_avg = 0.5*(R_p + R_s)
    return R_p, R_s, R_avg

def beamsplitter_transmittance(nus, h, theta, material=None, n=None, eps=None):
    '''
    n = real part of refractive index
    eps = real part of complex permittivity
    h = thickness of slab in m
    nus = list of frequencies in Hz
    theta = angle of incidence in degrees

    Returns:
    T_p = transmittance (fraction of incident power) for p-polarized light
    T_s = transmittance for s-polarized light
    T_avg = transmittance for unpolarized light
    '''
    if material is not None:
        n, eps = material_dict[material]
    elif n is None or eps is None:
        raise ValueError("You must provide both n and eps when material is not specified.")
    
    theta_rad = np.deg2rad(theta)
    T_p = oblique_slab_transmission_p(n, eps, h, nus, theta_rad)
    T_s = oblique_slab_transmission_s(n, eps, h, nus, theta_rad)
    T_avg = 0.5*(T_p + T_s)
    return T_p, T_s, T_avg

# will want to add more functions here for reflectance vs angle, etc.