import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# kappa fitting CSV obtained from https://automeris.io/

# === 1. Load CSV file ===
# Replace 'your_data.csv' with your actual CSV filename
data = pd.read_csv('rrC.csv', header=None)

# === 2. Extract columns ===
t = data[0].values
A = data[1].values

def decay(t, A0, kappa, C):
    return A0 * np.exp(-kappa * t / 2) + C

A0_guess = max(A) - min(A)
kappa_guess = 1.0 / (t[-1] - t[0])
C_guess = min(A)
p0 = [A0_guess, kappa_guess, C_guess]

params, covariance = curve_fit(decay, t, A, p0=p0)
A0_fit, kappa_fit, C_fit = params
kappa_MHz = kappa_fit * 1e3

plt.figure(figsize=(8,5))
plt.plot(t, A, 'bo', label='Digitized data')
plt.plot(t, decay(t, *params), 'r-', label=f'Fit: kappa = {kappa_MHz:.4f} MHz')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Exponential Decay Fit')
plt.legend()
plt.grid(True)
plt.show()