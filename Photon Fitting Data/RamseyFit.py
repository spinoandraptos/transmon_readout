from scipy.optimize import curve_fit
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# To adjust according to experiment
day ="2025-07-12"
pulse = "CLEAR"
# pulse = "rect"
qubit = "B"
start_time = datetime.strptime("17-09-00", "%H-%M-%S")
end_time   = datetime.strptime("17-38-00", "%H-%M-%S")
tlist = np.round(np.linspace(4, 1000, 30)).astype(int)

data_directory = f"C:\\Users\\admin\\Desktop\\cphase\\data\\{day}"
CLEAR_directory = "C:\\Users\\admin\Desktop\\cphase\\scripts\\CLEAR"

delta = 1e6 #1MHz virtual detuning

# T2 refers to T2 Echo
qubit_specs = {
    "qA": {
        "chi": 0.64e6,
        "kappa": 2.273e6,
        "T2": 21.5e-6
    },
    "qB": {
        "chi": 0.4e6,
        "kappa": 5.2125e6,
        "T2": 37e-6,
    },
    "qC": {
        "chi": 0.5e6,
        "kappa": 6.547e6,
        "T2": 6.5e-6, 
    }
}

# From CLEAR paper
"""
This expression accounts for:

1) How photons accumulate a time-varying phase shift on the qubit due to the Stark effect

2) How this phase shift decays over time as photons leave the cavity

3) The dephasing envelope from background decoherence (Γ₂)

So the qubit state acquires a nonlinear, time-dependent phase, and this is what's observed as a distorted Ramsey oscillation.
"""
def ramsey_model(t, n0, phi0, Gamma2, offset, kappa, delta, chi):

    chi *= 2 * np.pi
    kappa *= 2 * np.pi
    delta *= 2 * np.pi
    Gamma2 *= 2 * np.pi

    tau = (1 - np.exp(-(kappa + 2j * chi) * t)) / (kappa + 2j * chi)
    phase = -(Gamma2 + 1j * delta)* t + 1j * (phi0 - 2 * n0 * chi * tau)
    return 0.5 * (1 - np.imag(np.exp(phase)) + offset)


def extract_residual_photons(filepath):

    base, _ = filepath.rsplit('.', 1)  # split at last dot
    fit_filepath = f"{base}_ramsey_fit.jpeg"

    with h5py.File(filepath, 'r') as f:
        t_data = np.array(f["time_delay"]).flatten() * 1e-9  # in nanoseconds
        p_data = np.array(f["single_shot"])  # excited population [0,1]
        p_data_mean = p_data.mean(axis=0) 
        p_norm = (p_data_mean - np.min(p_data_mean)) / (np.max(p_data_mean) - np.min(p_data_mean))

    # Known constants from separate calibrations
    chi = qubit_specs[f"q{qubit}"]["chi"]
    kappa = qubit_specs[f"q{qubit}"]["kappa"]
    Gamma2 = 1 /qubit_specs[f"q{qubit}"]["T2"]

    # --- First fit: fit to all data initially ---
    popt_init, _ = curve_fit(
        lambda t, n0, phi0, Gamma2, offset: ramsey_model(t, n0, phi0, Gamma2, offset, kappa, delta, chi),
        t_data,
        p_norm,
        p0=[1.0, 0.0, 1 / qubit_specs[f"q{qubit}"]["T2"], 0.0],
        bounds=([0, -2*np.pi, 1/300e-6, -1.0], [15, 2*np.pi, 1/1e-6, 1.0])
    )

    # --- Compute residuals between model and data ---
    predicted = ramsey_model(t_data, *popt_init, kappa, delta, chi)
    residuals = np.abs(p_norm - predicted)

    # --- Define threshold for outlier removal ---
    threshold = 3 * np.std(residuals)

    # --- Create mask to filter inliers ---
    mask = residuals < threshold
    t_filtered = t_data[mask]
    p_filtered = p_norm[mask]

    # --- Final fit after removing outliers ---
    popt, pcov = curve_fit(
        lambda t, n0, phi0, Gamma2, offset: ramsey_model(t, n0, phi0, Gamma2, offset, kappa, delta, chi),
        t_filtered,
        p_filtered,
        p0=popt_init,  # use previous as initial guess
        bounds=([0, -2*np.pi, 1/300e-6, -1.0], [15, 2*np.pi, 1/1e-6, 1.0])
    )

    n0_fit, phi0_fit, T2_fit, ofs_fit = popt

    # --- Print estimated parameters ---
    # print("Estimated residual photon number (n0):", n0_fit)
    # print("Estimated phase (phi0):", phi0_fit)
    # print(f"Estimated T2: {1/T2_fit:.3e} s")
    # print(f"Estimated kappa: {kappa_fit/1e6:.3f} MHz")

    # --- Plot fit vs data ---
    plt.figure(figsize=(10, 4)) 
    plt.plot(t_data * 1e9, p_norm, 'bo', label='Data')
    plt.plot(t_data * 1e9, ramsey_model(t_data, *popt, kappa, delta, chi), 'r-', label='Fit')

    # Annotate plot with fitted values
    plt.text(
        0.98, 0.95,
        f"$n_0$ = {n0_fit:.4f}\n"
        f"$\\phi_0$ = {phi0_fit:.2f} rad\n"
        f"$T_2$ = {1/T2_fit*1e6:.1f} µs\n"
        f"$Ofs$ = {ofs_fit:.4f} µs\n",
        # f"$\\kappa$ = {kappa_fit/1e6:.2f} MHz",
        fontsize=15,
        color='black',
        ha='right',
        va='top',
        transform=plt.gca().transAxes
    )

    plt.xlabel("Ramsey Delay t (ns)")
    plt.ylabel("Excited State Population")
    plt.title("Ramsey Fit for Residual Photon Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fit_filepath, dpi=300)  # save figure with 300 dpi resolution
    plt.close() 

    return n0_fit

if __name__ == "__main__":

    photon_list = []

    file_entries = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".hdf5"):
            try:
                timestamp_str = filename.split("_")[0]
                file_time = datetime.strptime(timestamp_str, "%H-%M-%S")
                file_entries.append((file_time, filename))
            except ValueError:
                continue  # skip files that don't match the format

    # Sort by file_time (ascending)
    file_entries.sort()

    # Process files in order
    for file_time, filename in file_entries:
        if start_time <= file_time <= end_time:
            try:
                n = extract_residual_photons(os.path.join(data_directory, filename))
                photon_list.append(n)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        # --- Exponential decay model ---
    def photon_decay(t, n0, tau):
        return n0 * np.exp(-t / tau)

    # --- Fit ---
    popt, pcov = curve_fit(photon_decay, tlist/1e9, photon_list, p0=[photon_list[0], (tlist[-1] - tlist[0]) / 1e9 / 2])
    n0_fit, tau_fit = popt
    kappa_fit = 1 / tau_fit

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(tlist, photon_list, 'bo', label="Data")  # plot in ns
    plt.plot(tlist, photon_decay(tlist/1e9, *popt), 'r-', label="Fit")

    # Add fitted parameters as text
    textstr = (
        f"$n_0$ = {n0_fit:.3f}\n"
        f"$\\tau$ = {tau_fit * 1e6 :.2f} µs\n"
        f"$\\kappa$ = {kappa_fit / 1e6 :.2f} MHz"
    )
    plt.text(
        0.95, 0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.xlabel("Time (ns)")
    plt.ylabel("Residual Photon Number")
    plt.title("Photon Decay in Readout Resonator")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAR_directory, f"{qubit}_n0_{pulse}.png"), dpi=300)  # save figure with 300 dpi resolution
    plt.show()
    
