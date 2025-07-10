from scipy.optimize import curve_fit
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# To adjust according to experiment
qubit = "B"
start_time = datetime.strptime("14-14-00", "%H-%M-%S")
end_time   = datetime.strptime("14-31-00", "%H-%M-%S")
tlist = np.round(np.linspace(4, 1000, 30)).astype(int)


relative_dir = "Test"  # working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(script_dir, relative_dir)
CLEAR_directory = ""

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
def ramsey_model(t, n0, phi0, Gamma2, kappa, delta, chi):

    chi *= 2 * np.pi
    kappa *= 2 * np.pi
    delta *= 2 * np.pi
    Gamma2 *= 2 * np.pi

    tau = (1 - np.exp(-(kappa + 2j * chi) * t)) / (kappa + 2j * chi)
    phase = -(Gamma2 + 1j * delta)* t + 1j * (phi0 - 2 * n0 * chi * tau)
    return 0.5 * (1 - np.imag(np.exp(phase)))


def extract_residual_photons(filepath):
    # filepath = f"Photon Fitting Data/q{qubit}.hdf5"
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

    # t_data in nanosecon ds, p_data is measured excited state probability
    popt, pcov = curve_fit(
        lambda t, n0, phi0, T2: ramsey_model(t, n0, phi0, T2, kappa, delta, chi),
        t_data,
        p_norm,
        p0=[1.0, 0, 1/qubit_specs[f"q{qubit}"]["T2"]],  # Initial guesses: n0, phi0,
        bounds=([0, -2*np.pi, 1/300e-6], [15, 2*np.pi, 1/1e-6]),  # example
    )

    n0_fit, phi0_fit, T2_fit = popt

    # --- Print estimated parameters ---
    print("Estimated residual photon number (n0):", n0_fit)
    print("Estimated phase (phi0):", phi0_fit)
    print(f"Estimated T2: {1/T2_fit:.3e} s")
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
        f"$T_2$ = {1/T2_fit*1e6:.1f} µs\n",
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

    for filename in os.listdir(data_directory):
        if filename.endswith(".hdf5"):
            try:
                # Extract HH-MM-SS from filename
                timestamp_str = filename.split("_")[0]
                file_time = datetime.strptime(timestamp_str, "%H-%M-%S")

                # Check if within desired range
                if start_time <= file_time <= end_time:
                    n = extract_residual_photons(os.path.join(data_directory, filename))
                    photon_list.append(n)

            except Exception as e:
                print(f"Skipping {filename}: {e}")

    plt.figure()
    plt.plot(tlist, photon_list)
    plt.xlabel("Time (ns)")
    plt.ylabel("Residual Photon Number")
    plt.title("Residual Photon Number vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CLEAR_directory, "n0.png"), dpi=300)  # save figure with 300 dpi resolution
    plt.show()
    
