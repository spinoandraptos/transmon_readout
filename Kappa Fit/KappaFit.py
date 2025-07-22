import numpy as np
import h5py
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import hilbert

sample_rate = 1e9       # 1 GSa/s
start_idx = 1280         # Start time index of decay in trace

rr = 'rrC'

adc_idx = {
    'rrA': 'adc3',
    'rrB': 'adc2',
    'rrC': 'adc',

}

def exp_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B

filepath = f"Kappa Fitting Data/{rr}.hdf5"
base, _ = filepath.rsplit('.', 1)  # split at last dot
fit_filepath = f"{base}_kappa_fit.png"

with h5py.File(filepath, 'r') as f:
    adc_raw = f[adc_idx[rr]]
    adc = np.mean(adc_raw, axis=0)
    adc = adc - np.mean(adc)
    N_shots, N_samples = adc_raw.shape
    t = np.arange(N_samples) / sample_rate 

    analytic_signal = hilbert(adc)         # raw_trace: 1D real-valued
    envelope = np.abs(analytic_signal)  

    envelope_smooth = savgol_filter(envelope, window_length=50, polyorder=4)

    plt.figure(figsize=(10, 5))

    # Plot original raw ADC trace
    plt.plot(t * 1e6, adc, label="Original Trace", alpha=0.5)

    # Plot smoothed envelope
    plt.plot(t * 1e6, envelope_smooth, label="Smoothed Envelope", linewidth=2)

    # Plot exponential fit on top (only on the second ringdown region)
    t_fit = t[start_idx:]
    env_fit = envelope_smooth[start_idx:]

    # Fit
    A0 = env_fit[0] - env_fit[-1]
    tau0 = (t_fit[-1] - t_fit[0]) / 2
    B0 = env_fit[-1]
    p0 = [A0, tau0, B0]
    params, _ = curve_fit(exp_decay, t_fit, env_fit, p0=p0)
    A_fit, tau_fit, B_fit = params

    # Extract kappa
    kappa = 1/tau_fit * 2
    kappa_MHz = kappa / (2 * np.pi * 1e6)

    # Plot fit
    t_fit_us = t_fit * 1e6
    fit_curve = exp_decay(t_fit, *params)
    plt.plot(t_fit_us, fit_curve, 'r--', linewidth=2,
            label=f'Fit: κ / 2π = {kappa_MHz:.3f} MHz, τ = {1/kappa*1e6:.2f} µs')

    # Labels
    plt.xlabel("Time (µs)")
    plt.ylabel("Signal Amplitude")
    plt.title("Resonator Ringdown with Fitted Decay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fit_filepath, dpi=300)

# # kappa fitting CSV obtained from https://automeris.io/

# # === 1. Load CSV file ===
# # Replace 'your_data.csv' with your actual CSV filename
# data = pd.read_csv('rrC.csv', header=None)

# # === 2. Extract columns ===
# t = data[0].values
# A = data[1].values

# def decay(t, A0, kappa, C):
#     return A0 * np.exp(-kappa * t / 2) + C

# A0_guess = max(A) - min(A)
# kappa_guess = 1.0 / (t[-1] - t[0])
# C_guess = min(A)
# p0 = [A0_guess, kappa_guess, C_guess]

# params, covariance = curve_fit(decay, t, A, p0=p0)
# A0_fit, kappa_fit, C_fit = params
# kappa_MHz = kappa_fit * 1e3

# plt.figure(figsize=(8,5))
# plt.plot(t, A, 'bo', label='Digitized data')
# plt.plot(t, decay(t, *params), 'r-', label=f'Fit: kappa = {kappa_MHz:.4f} MHz')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Exponential Decay Fit')
# plt.legend()
# plt.grid(True)
# plt.show()
