import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

filename = "Power Fitting Data/rrA.hdf5"

# --- Lorentzian peak model ---
def lorentzian(f, A, f0, gamma, C):
    return A * gamma**2 / ((f - f0)**2 + gamma**2) + C

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
    # Initial guess: A, f0, gamma, C
    A_guess = np.max(y) - np.min(y)
    f0_guess = freq[np.argmax(y)]
    gamma_guess = (freq[-1] - freq[0]) / 20
    C_guess = np.min(y)
    try:
        popt, _ = curve_fit(lorentzian, freq, y,
                            p0=[A_guess, f0_guess, gamma_guess, C_guess])
        fit_freqs.append(popt[1])  # Extract f0
    except RuntimeError:
        fit_freqs.append(np.nan)  # Mark bad fits

fit_freqs = np.array(fit_freqs)
valid_mask = ~np.isnan(fit_freqs)
amp_valid = amp[valid_mask]
delta_f = fit_freqs[valid_mask] - fit_freqs[valid_mask][0]

# --- Fit linear region (first N points) ---
N_fit = 10
slope, intercept, *_ = linregress(amp_valid[:N_fit], delta_f[:N_fit])
linear_fit = slope * amp_valid + intercept
residual = np.abs(delta_f - linear_fit)

# --- Define deviation threshold (e.g., 1 MHz) ---
threshold = 3 * np.std(delta_f[:N_fit] - linear_fit[:N_fit])  # 3σ criterion
idx_exceed = np.where(residual > threshold)[0]

# --- Determine max safe amplitude ---
if len(idx_exceed) > 0:
    max_safe_idx = idx_exceed[0] - 1
    max_safe_amp = amp_valid[max_safe_idx]
else:
    max_safe_amp = amp_valid[-1]

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(amp_valid, delta_f / 1e6, 'o-', label="Measured Δf_q")
plt.plot(amp_valid, linear_fit / 1e6, '--', label="Linear fit")
plt.text(max_safe_amp, max(delta_f)/1e6 * 0.5,
         f"Safe limit:\n{max_safe_amp:.3f}",
         color="red", ha="right")
plt.axvline(max_safe_amp, color='red', linestyle=':', label="Max safe amplitude")
plt.xlabel("Readout Amplitude")
plt.ylabel("Qubit Stark Shift Δf_q (MHz)")
plt.title("AC Stark Shift vs Readout Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
