from scipy.optimize import curve_fit
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# To adjust according to experiment
day ="2025-07-21"
# pulse = "CLEAR"
pulse = "rect"
start_time = datetime.strptime("16-10-00", "%H-%M-%S")
end_time   = datetime.strptime("16-37-00", "%H-%M-%S")
tlist = np.round(np.linspace(4, 1000, 30)).astype(int) 

data_directory = f"C:\\Users\\qcrew\\Documents\\jon\\cheddar\\data\{day}"
CLEAR_directory = "C:\\Users\\qcrew\\Documents\\jon\\cheddar\\scripts\\CLEAR"

delta = 1e6 #1MHz virtual detuning
N0 = 22
FIT_FRAC = 0.8 # Fraction of points to use for exp fitting (starting from back)

# T2 refers to T2 Echo
qubit_specs = {
    "chi": 1.05e6,
    "kappa": 0.336e6,
    "T2": 26.9e-6
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

# --- Exponential decay model ---
def photon_decay(t, n0, tau):
    return n0 * np.exp(-t / tau)

def extract_residual_photons(filepath, count):

    base, _ = filepath.rsplit('.', 1)  # split at last dot
    fit_filepath = f"{base}_ramsey_fit.jpeg"

    with h5py.File(filepath, 'r') as f:
        t_data = np.array(f["time_delay"]).flatten() * 1e-9  # in nanoseconds
        p_data = np.array(f["single_shot"])  # excited population [0,1]
        p_data_mean = p_data.mean(axis=0) 
        p_norm = (p_data_mean - np.min(p_data_mean)) / (np.max(p_data_mean) - np.min(p_data_mean))

    # Known constants from separate calibrations
    chi = qubit_specs["chi"]
    kappa = qubit_specs["kappa"]
    Gamma2 = 1 /qubit_specs["T2"]
    phi_guess = np.arcsin(2 * [p_norm[0] - 0.5])[0]
    
    n0_guess = photon_decay(tlist[count]/1e9, N0, 1/kappa)
    # print(n0_guess)

    # --- First fit: fit to all data initially ---
    popt_init, _ = curve_fit(
        lambda t, n0, phi0, Gamma2, offset: ramsey_model(t, n0, phi0, Gamma2, offset, kappa, delta, chi),
        t_data,
        p_norm,
        p0=[n0_guess, phi_guess, Gamma2, 0.0],
        bounds=([0, -2*np.pi, 1/1e-3, -1.0], [N0, 2*np.pi, 1/1e-6, 1.0])
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
    # weights = 1/(p_filtered + 0.02)

    # --- Final fit after removing outliers ---
    popt, pcov = curve_fit(
        lambda t, n0, phi0, Gamma2, offset: ramsey_model(t, n0, phi0, Gamma2, offset, kappa, delta, chi),
        t_filtered,
        p_filtered,
        p0=popt_init,  # use previous as initial guess
        bounds=([0, -2*np.pi, 1/1e-3, -1.0], [N0, 2*np.pi, 1/1e-6, 1.0]),
        # sigma=weights
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
                if start_time < file_time < end_time:
                    file_entries.append((file_time, filename))
            except ValueError:
                continue  # skip files that don't match the format

    # Sort by file_time (ascending)
    file_entries.sort()
    sorted_filenames = [filename for _, filename in file_entries]
   

    count = 0
    photon_list = []
    group_filenames = sorted_filenames
    for filename in group_filenames:
        try:
            full_path = os.path.join(data_directory, filename)
            n = extract_residual_photons(full_path, count)
            photon_list.append(n)
            count += 1
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
    photon_list = np.array(photon_list)
    tlist = np.array(tlist)
    fraction = FIT_FRAC  # fraction of data at tail for fitting
    start_idx = int(len(tlist) * (1 - fraction))
    # Use tail data for fitting only
    t_fit = tlist[start_idx:] / 1e9  # convert to seconds if needed
    photon_fit = photon_list[start_idx:]
    # Fit exponential decay to tail data
    try:
        popt, pcov = curve_fit(photon_decay, t_fit, photon_fit,
                              p0=[photon_fit[0], (t_fit[-1] - t_fit[0]) / 2])
        n0_fit, tau_fit = popt
        kappa_fit = 1 / tau_fit
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = [np.nan, np.nan]
    # Calculate residuals over full data
    residuals = np.abs(photon_list - photon_decay(tlist / 1e9, *popt))
    threshold = 3 * np.std(residuals)
    outliers_mask = residuals > threshold
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(tlist, photon_list, 'bo', label='Data')
    plt.plot(tlist, photon_decay(tlist / 1e9, *popt), 'r-', label='Fit (extrapolated)')
    # Highlight outliers on full data
    # plt.plot(tlist[outliers_mask], photon_list[outliers_mask], 'rx', markersize=10, label='Outliers')
    
    # Fitted parameters as text
    textstr = (
        f"$n_0$ = {n0_fit:.3f}\n"
        f"$\\tau$ = {tau_fit * 1e6 :.2f} µs\n"
        f"$\\kappa$ = {kappa_fit / 1e6 :.2f} MHz"
    )
    plt.text(0.95, 0.95, textstr,
            transform=plt.gca().transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.xlabel("Time (ns)")
    plt.ylabel("Residual Photon Number")
    plt.title(f"Qubit {pulse.upper()} - Photon Decay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(CLEAR_directory, f"qubit_{pulse}_n0.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved plot for qubit {pulse} → {fig_path}")