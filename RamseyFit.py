from scipy.optimize import curve_fit
import h5py
import numpy as np
import matplotlib.pyplot as plt

filepath = "your_ramsey_data.h5"

with h5py.File(filepath, 'r') as f:
    t_data = np.array(f["t_delay"])  # in seconds
    p_data = np.array(f["excited_pop"])  # already normalized, or [0,1]

def ramsey_model(t, n0, phi0, kappa, chi, Gamma2):
    tau = (1 - np.exp(-(kappa + 2j * chi) * t)) / (kappa + 2j * chi)
    phase = -Gamma2 * t + 1j * (phi0 - 2 * n0 * chi * tau)
    return 0.5 * (1 - np.imag(np.exp(phase)))

# Known constants from separate calibrations
chi = 2 * np.pi * 1e6  # Hz
kappa = 1 / 300e-9     # s^-1
Gamma2 = 1 / 30e-6     # s^-1 -> T2

# t_data in seconds, p_data is measured excited state probability
popt, pcov = curve_fit(
    lambda t, n0, phi0: ramsey_model(t, n0, phi0, kappa, chi, Gamma2),
    t_data,
    p_data,
    p0=[0.1, 0]  # Initial guesses: n0, phi0
)

n0_fit, phi0_fit = popt
print("Estimated residual photon number:", n0_fit)
