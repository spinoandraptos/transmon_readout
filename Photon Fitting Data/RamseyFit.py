from scipy.optimize import curve_fit
import h5py
import numpy as np
import matplotlib.pyplot as plt

qubit = "A"

filepath = f"Photon Fitting Data/q{qubit}.hdf5"
base, _ = filepath.rsplit('.', 1)  # split at last dot
fit_filepath = f"{base}_ramsey_fit.jpeg"

delta = 2 * np.pi * 1e6 #1MHz virtual detuning

# T2 refers to T2 Echo
qubit_specs = {
    "qA": {
        "chi": 0.64e6,
        "kappa": 2.273e6,
        "T2": 110.2e-6, # Adjust this to fit to measurement
    },
    "qB": {
        "chi": 0.4e6,
        "kappa": 5.2125e6,
        "T2": 250.2e-6, # Adjust this to fit to measurement
    },
    "qC": {
        "chi": 0.5e6,
        "kappa": 6.547e6,
        "T2": 6.5e-6, # Adjust this to fit to measurement
    }
}

with h5py.File(filepath, 'r') as f:
    t_data = np.array(f["time_delay"]).flatten() * 1e-9  # in nanoseconds
    p_data = np.array(f["single_shot"])  # excited population [0,1]
    p_data_mean = p_data.mean(axis=0) 
    p_norm = (p_data_mean - np.min(p_data_mean)) / (np.max(p_data_mean) - np.min(p_data_mean))


# From CLEAR paper
"""
This expression accounts for:

1) How photons accumulate a time-varying phase shift on the qubit due to the Stark effect

2) How this phase shift decays over time as photons leave the cavity

3) The dephasing envelope from background decoherence (Γ₂)

So the qubit state acquires a nonlinear, time-dependent phase, and this is what's observed as a distorted Ramsey oscillation.
"""
def ramsey_model(t, n0, phi0, delta, kappa, chi, Gamma2):
    tau = (1 - np.exp(-(kappa + 2j * chi) * t)) / (kappa + 2j * chi)
    phase = -(Gamma2 + 1j * delta)* t + 1j * (phi0 - 2 * n0 * chi * tau)
    return 0.5 * (1 - np.imag(np.exp(phase)))

# Known constants from separate calibrations
chi = 2 * np.pi * qubit_specs[f"q{qubit}"]["chi"]
kappa = 2 * np.pi * qubit_specs[f"q{qubit}"]["kappa"]
Gamma2 = 2 * np.pi * 1 /qubit_specs[f"q{qubit}"]["T2"]

# t_data in nanosecon ds, p_data is measured excited state probability
popt, pcov = curve_fit(
    lambda t, n0, phi0: ramsey_model(t, n0, phi0, delta, kappa, chi, Gamma2),
    t_data,
    p_norm,
    p0=[1.0, 0],  # Initial guesses: n0, phi0,
    bounds=([0, -2*np.pi], [15, 2*np.pi]),  # example
)

n0_fit, phi0_fit = popt
print("Estimated residual photon number:", n0_fit)
print("Estimated phase:", phi0_fit)

# Plot fit vs data
plt.figure(figsize=(10, 4)) 
plt.plot(t_data * 1e9, p_norm, 'bo', label='Data')
plt.plot(t_data * 1e9, ramsey_model(t_data, *popt, delta, kappa, chi, Gamma2), 'r-', label='Fit')
plt.text(
    0.98, 0.95,  # x,y in axes fraction (0 to 1)
    f"$n_0$ = {n0_fit:.4f}, $\\phi_0$ = {phi0_fit:.2f} rad",
    fontsize=15,
    color='black',
    ha='right',
    va='top',
    transform=plt.gca().transAxes  # coordinate system relative to axes
)
plt.xlabel("Ramsey Delay t (ns)")
plt.ylabel("Excited State Population")
plt.title("Ramsey Fit for Residual Photon Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fit_filepath, dpi=300)  # save figure with 300 dpi resolution

plt.show()