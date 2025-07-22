import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os

data_directory = "/home/spinoandraptos/Documents/CQT/Experiments/Power Fitting Data"

# --- Lorentzian peak model ---
def lorentzian(f, A, f0, gamma, C):
    return A * gamma**2 / ((f - f0)**2 + gamma**2) + C

def calibrate_power(filename):
    base, _ = filename.rsplit('.', 1)  # split at last dot
    fit_filepath = f"{base}_power_calibration.png"
    annotation_filepath = f"{base}_amplitude_annotation.png"

    with h5py.File(filename, "r") as f:
        freq = f["qubit_frequency"][:]        # Shape: (N_freq,)
        amp = f["readout_amp"][:]        # Shape: (N_amp,)
        data = f["single_shot"][:]                 # Shape: (N_amp, N_freq), or use "Q" if needed

    # --- Average over repetitions ---
    data_avg = np.mean(data, axis=0)  # shape: (20, 101)

    # --- Fit each trace to get precise qubit frequency ---
    fit_freqs = []
    for i in range(len(amp)):
        y = data_avg[i]
        y = gaussian_filter1d(y, sigma=1)
        # Initial guess: A, f0, gamma, C
        A_guess = np.max(y) - np.min(y)
        f0_guess = freq[np.argmax(y)]
        gamma_guess = (freq[-1] - freq[0]) / 20
        C_guess = np.min(y)
        try:
            popt, _ = curve_fit(
                lorentzian, freq, y,
                p0=[A_guess, f0_guess, gamma_guess, C_guess],
                bounds=([0, freq[0], 0, -np.inf], [np.inf, freq[-1], np.inf, np.inf])
)
            fit_freqs.append(popt[1])  # Extract f0
        except RuntimeError:
            fit_freqs.append(np.nan)  # Mark bad fits

    fit_freqs = np.array(fit_freqs)
    valid_mask = ~np.isnan(fit_freqs)
    amp_valid = amp[valid_mask]
    delta_f = fit_freqs[valid_mask] - fit_freqs[valid_mask][0]

    N_fit = 5
    slope, intercept, *_ = linregress(amp_valid[:N_fit], delta_f[:N_fit])
    expected = slope * amp_valid + intercept
    residual = np.abs(delta_f - expected)
    threshold = 3 * np.std(residual)  # 1 MHz
    idx_exceed = np.where(residual > threshold)[0]
    if len(idx_exceed) > 0:
        max_safe_idx = idx_exceed[0] - 1
        max_safe_amp = amp_valid[max_safe_idx]
    else:
        max_safe_amp = amp_valid[-1]


    plt.figure(figsize=(8, 6))
    plt.imshow(data_avg, aspect='auto', origin='lower',
            extent=[freq[0]/1e6, freq[-1]/1e6, amp[0], amp[-1]],
            cmap='viridis')
    plt.colorbar(label='Signal')
    plt.xlabel("Qubit Frequency (MHz)")
    plt.ylabel("Readout Amplitude")
    plt.title("2D Scan: AC Stark Shift")
    plt.axhline(max_safe_amp, color='red', linestyle='--', label="Max safe amplitude")
    plt.legend()
    plt.savefig(annotation_filepath, dpi=300)
    
    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(amp_valid, delta_f / 1e6, 'o-', label="Measured Δf_q")
    plt.plot(amp_valid, expected / 1e6, '--', color='orange', label="Linear fit (first N points)")
    plt.axvline(max_safe_amp, color='red', linestyle=':', label="Max safe amplitude")
    plt.text(max_safe_amp, max(delta_f)/1e6 * 0.5,
            f"Safe limit:\n{max_safe_amp:.3f}",
            color="red", ha="right")
    plt.xlabel("Readout Amplitude")
    plt.ylabel("Qubit Stark Shift Δf_q (MHz)")
    plt.title("AC Stark Shift vs Readout Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fit_filepath, dpi=300)

if __name__ == "__main__":
    for filename in os.listdir(data_directory):
        if filename.endswith("rrJon.hdf5"):
            try:
                calibrate_power(os.path.join(data_directory, filename))
            except Exception as e:
                print(f"Skipping {filename}: {e}")
