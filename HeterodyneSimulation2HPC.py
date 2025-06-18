from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import json

### To transfer: scp /home/spinoandraptos/Documents/CQT/Experiments/HeterodyneSimulationHPC.py e0968844@atlas8.nus.edu.sg:/home/svu/e0968844

# Define circuit parameters

# Define number of fock states in resonator
N = 20

# Circuit parameters
wr = 7.062 * 2 * np.pi      # Resonator frequency (7.062 GHz)[from reference]
wq = 5.092 * 2 * np.pi      # Qubit frequency (5.092 GHz)[from reference]
delta = abs(wr - wq)        # Detuning between qubit and resonator (1.97 GHz)[from reference]
k = 0.00535 * 2 * np.pi     # Resonator decay rate (5.35 MHz)[from reference]
g = 0.1029 * 2 * np.pi      # Coupling strength (102.9 MHz)[from reference]
gamma_1 = 0   # Qubit decay (1 MHz) [self-defined, not from reference]
gamma_2 = 0   # Qubit dephase (1 MHz) [self-defined, not from reference]

wr_d = wr - g**2 / delta    # Dressed resonator frequency (9.9997 GHz)
wq_d = wq + g**2 / delta    # Dressed qubit frequency (5.0003 GHz)
wd = wr

alpha = 0.157 * 2 * np.pi   # Anharmonity (157 MHz) [Chosen to align results]
K = alpha * (g/delta)**4      # Kerr self-nonlinearity (8.18 KHz) [Derived from Yvonne reference]
chi = 2 * alpha * (g/delta)**2 # Dispersive shift, cross non-linearity (5.378 MHz) [Derived from Yvonne reference]

n_crit = (delta / 2*g)**2  # Critical photon number (16)

# Define quantum operators
a = tensor(destroy(N), qeye(2))    # Resonator lowering operator
q = tensor(qeye(N), destroy(2))    # Qubit lowering operator
# Define qubit Pauli operators in composite space
sx = tensor(qeye(N), sigmax())
sy = tensor(qeye(N), sigmay())
sz = tensor(qeye(N), sigmaz())

# Base Hamiltonian without drive (lab frame):
H0 = (wr - wd + chi * sz) * a.dag() * a + (wq -wd)/2 * sz
drive_power = 4.984375000000001e-05 * 2 * np.pi

# Define collapse operators
c_ops = [np.sqrt(k) * a]  # Resonator decay

# Define measurement operators
e_ops = [a.dag() * a, a]  # Photon number in resonator and qubit state, resonator field

# Initial state: resonator in vacuum, qubit in ground state
psi0 = tensor(basis(N, 0), basis(2,0))  # Vacuum state for resonator, ground state for qubit
psi1 = tensor(basis(N, 0), basis(2,1))  # Vacuum state for resonator, excited state for qubit
psi4 = tensor(coherent(N, 4), basis(2,0))  # Coherent state (eigenvalue 4) for resonator, ground state for qubit
psis = tensor(basis(N,0), (basis(2,0) + basis(2,1)).unit())  # Superposition of ground and excited state for qubit

# Time evolution parameters
tlist = np.linspace(0, 800, 8000)  # Time from 0 to 800 ns

# Full CLEAR signal and photon response using optimal ring-up and ring-down times
drive_power = 4.984375000000001e-05 * 2 * np.pi

ringup1_norm = 11.58332303322993
ringdown1_norm = 1.6156046841062026
ringup1_amp = np.sqrt(ringup1_norm * drive_power)  # Amplitude of the ring-up pulse
ringdown1_amp = np.sqrt(ringdown1_norm * drive_power)  # Amplitude of the ring-down pulse

ringup2_norm = 12.242809267406063
drive_norm = 2.0615066563271975
ringdown2_norm = 14.632117912321677
ringup2_amp = np.sqrt(ringup2_norm * drive_power)  # Amplitude of the ring-up pulse
drive_amp = np.sqrt(drive_norm * drive_power)  # Amplitude of the drive pulse
ringdown2_amp = -np.sqrt(ringdown2_norm * drive_power)  # Amplitude of the ring-down pulse

optimal_ringup1_time = 47.6257865352412
optimal_ringdown1_time = 90.36540617777688
optimal_ringup2_time = 20.668371079660396
optimal_ringdown2_time = 44.48693108893437


def CLEAR_pulse(t, args):
    if t<=optimal_ringup1_time:
        return ringup1_amp 
    elif t <= optimal_ringup1_time + optimal_ringdown1_time:
        return ringdown1_amp
    elif t <= 500:
        return drive_amp
    elif t <= 500 + optimal_ringdown2_time:
        return ringdown2_amp 
    elif t <= 500 + optimal_ringdown2_time + optimal_ringup2_time:
        return ringup2_amp
    else:   
        return 0.0

def drive_pulse(t, args):
    return drive_amp if t <= 500 else 0  # Pulse from 100 ns to 200 ns

HD_CLEAR = [(a + a.dag()), CLEAR_pulse]
H_CLEAR = [H0, HD_CLEAR]

HD_drive = [(a + a.dag()), drive_pulse]
H_drive = [H0, HD_drive]

ntrajs = 3000
e_ops = [a + a.dag(), -1j * (a - a.dag())]  # Measurement operators for IQ points
c_ops = [np.sqrt(k) * a, np.sqrt(gamma_1) * q, np.sqrt(gamma_2) * sz]  # Resonator decay and qubit decay
sc_ops = [np.sqrt(k) * a] # Measuring the channel of the resonator field

dt = tlist[1] - tlist[0]

# Simulate a single heterodyne measurement trajectory
result_g = smesolve(
    H_CLEAR,
    psi0,
    tlist,
    ntraj=ntrajs,
    c_ops=c_ops,
    e_ops=e_ops,
    sc_ops=sc_ops,  # heterodyne detection
    heterodyne=True,
    options={'store_measurement':True, "map": "parallel", 'progress_bar':'enhanced'}
)

result_e = smesolve(
    H_CLEAR,
    psi1,
    tlist,
    ntraj=ntrajs,
    c_ops=c_ops,
    e_ops=e_ops,
    sc_ops=sc_ops,  # heterodyne detection
    heterodyne=True,
    options={'store_measurement':True, "map": "parallel", 'progress_bar':'enhanced'}
)

plt.figure(figsize=(12, 6))

plt.plot(tlist[1:], np.mean(result_g.measurement, axis=0)[0, 0, :].real/ np.sqrt(dt), "g", lw=1, alpha=0.5, label='I (In-phase, Qubit Ground)')
plt.plot(tlist[1:], np.mean(result_g.measurement, axis=0)[0, 1, :].real/ np.sqrt(dt), "y", lw=1, alpha=0.5, label='Q (Quadrature, Qubit Ground)')
plt.plot(tlist[1:], np.mean(result_e.measurement, axis=0)[0, 0, :].real/ np.sqrt(dt), "b", lw=1, alpha=0.5, label='I (In-phase, Qubit Energised)')
plt.plot(tlist[1:], np.mean(result_e.measurement, axis=0)[0, 1, :].real/ np.sqrt(dt), "r", lw=1, alpha=0.5, label='Q (Quadrature, Qubit Energised)')

plt.plot(tlist, result_g.expect[0]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_g.expect[1]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_e.expect[0]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_e.expect[1]/ np.sqrt(dt), lw=2)

plt.xlabel("Time (μs)")
plt.ylabel("Signal (a.u.)")
plt.title("Heterodyne Measurement Record")
plt.legend()
plt.grid(True)
plt.savefig("CLEAR_Heterodyne_3000_ideal_qubit.png")  # Save to file

I_g = list()
Q_g = list()
I_e = list()
Q_e = list()

for m in result_g.measurement:
    I = m[0, 0, 1500:5000]
    Q = m[0, 1, 1500:5000]
    I_int = np.sum(I) * dt
    Q_int = np.sum(Q) * dt
    I_g.append(I_int)
    Q_g.append(Q_int)

for m in result_e.measurement:
    I = m[0, 0, 1500:5000]
    Q = m[0, 1, 1500:5000]
    I_int = np.sum(I) * dt
    Q_int = np.sum(Q) * dt
    I_e.append(I_int)
    Q_e.append(Q_int)

IQ_g = np.vstack([I_g, Q_g]).T
IQ_e = np.vstack([I_e, Q_e]).T

# Stack all IQs and labels
IQ_all = np.vstack([IQ_g, IQ_e])
labels = np.array([0]*len(IQ_g) + [1]*len(IQ_e))  # 0 for |g⟩, 1 for |e⟩

# Train LDA
lda = LinearDiscriminantAnalysis()
lda.fit(IQ_all.real, labels)
preds = lda.predict(IQ_all.real)

coef = lda.coef_[0]      # [w1, w2]
intercept = lda.intercept_[0]  # w0

w1, w2 = coef
y_vals = -(intercept + w1 * IQ_all) / w2

# Readout fidelity
fidelity = accuracy_score(labels, preds)

# Mean vectors
mu_g = np.mean(IQ_g, axis=0)
mu_e = np.mean(IQ_e, axis=0)

# Difference vector (signal direction)
d = mu_e - mu_g
d_unit = d / np.linalg.norm(d)

# Project all IQ points onto signal axis
proj_g = IQ_g @ d_unit
proj_e = IQ_e @ d_unit

# Compute noise variance (assume equal noise for both classes)
sigma2 = 0.5 * (np.var(proj_g) + np.var(proj_e))

# Compute SNR
snr = (np.linalg.norm(mu_e - mu_g) ** 2) / sigma2

figures = {"fidelity": fidelity, "SNR": snr}
with open("CLEAR_figures_ideal_qubit.json", "w") as f:
    json.dump(figures, f)

plt.figure(figsize=(6, 6))

plt.scatter(I_g, Q_g, color='blue', alpha=0.5, label='|g⟩ shots (Qubit Relaxation)')
plt.scatter(I_e, Q_e, color='red', alpha=0.5, label='|e⟩ shots (Qubit Relaxation)')
plt.plot(IQ_all, y_vals, 'k', label='Decision Boundary (Qubit Relaxation)')
plt.xlabel("I (integrated)")
plt.ylabel("Q (integrated)")
plt.title("IQ Distribution for Qubit States")
plt.legend()
plt.grid(True)
plt.ylim(min(Q_g + Q_e) - 5,max(Q_g + Q_e) + 5)
plt.xlim(min(I_g + I_e) - 5,max(I_g + I_e) + 5)
plt.savefig("CLEAR_IQ_3000_ideal_qubit.png")  # Save to file

# Simulate a single heterodyne measurement trajectory
result_g = smesolve(
    H_drive,
    psi0,
    tlist,
    ntraj=ntrajs,
    c_ops=c_ops,
    e_ops=e_ops,
    sc_ops=sc_ops,  # heterodyne detection
    heterodyne=True,
    options={'store_measurement':True, "map": "parallel", 'progress_bar':'enhanced'}
)

result_e = smesolve(
    H_drive,
    psi1,
    tlist,
    ntraj=ntrajs,
    c_ops=c_ops,
    e_ops=e_ops,
    sc_ops=sc_ops,  # heterodyne detection
    heterodyne=True,
    options={'store_measurement':True, "map": "parallel", 'progress_bar':'enhanced'}
)

plt.figure(figsize=(12, 6))

plt.plot(tlist[1:], np.mean(result_g.measurement, axis=0)[0, 0, :].real/ np.sqrt(dt), "g", lw=1, alpha=0.5, label='I (In-phase, Qubit Ground)')
plt.plot(tlist[1:], np.mean(result_g.measurement, axis=0)[0, 1, :].real/ np.sqrt(dt), "y", lw=1, alpha=0.5, label='Q (Quadrature, Qubit Ground)')
plt.plot(tlist[1:], np.mean(result_e.measurement, axis=0)[0, 0, :].real/ np.sqrt(dt), "b", lw=1, alpha=0.5, label='I (In-phase, Qubit Energised)')
plt.plot(tlist[1:], np.mean(result_e.measurement, axis=0)[0, 1, :].real/ np.sqrt(dt), "r", lw=1, alpha=0.5, label='Q (Quadrature, Qubit Energised)')

plt.plot(tlist, result_g.expect[0]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_g.expect[1]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_e.expect[0]/ np.sqrt(dt), lw=2)
plt.plot(tlist, result_e.expect[1]/ np.sqrt(dt), lw=2)

plt.xlabel("Time (μs)")
plt.ylabel("Signal (a.u.)")
plt.title("Heterodyne Measurement Record")
plt.legend()
plt.grid(True)
plt.savefig("Rect_Heterodyne_3000_ideal_qubit.png")  # Save to file

I_g = list()
Q_g = list()
I_e = list()
Q_e = list()

for m in result_g.measurement:
    I = m[0, 0, 1500:5000]
    Q = m[0, 1, 1500:5000]
    I_int = np.sum(I) * dt
    Q_int = np.sum(Q) * dt
    I_g.append(I_int)
    Q_g.append(Q_int)

for m in result_e.measurement:
    I = m[0, 0, 1500:5000]
    Q = m[0, 1, 1500:5000]
    I_int = np.sum(I) * dt
    Q_int = np.sum(Q) * dt
    I_e.append(I_int)
    Q_e.append(Q_int)

IQ_g = np.vstack([I_g, Q_g]).T
IQ_e = np.vstack([I_e, Q_e]).T

# Stack all IQs and labels
IQ_all = np.vstack([IQ_g, IQ_e])
labels = np.array([0]*len(IQ_g) + [1]*len(IQ_e))  # 0 for |g⟩, 1 for |e⟩

# Train LDA
lda = LinearDiscriminantAnalysis()
lda.fit(IQ_all.real, labels)
preds = lda.predict(IQ_all.real)

coef = lda.coef_[0]      # [w1, w2]
intercept = lda.intercept_[0]  # w0

w1, w2 = coef
y_vals = -(intercept + w1 * IQ_all) / w2

# Readout fidelity
fidelity = accuracy_score(labels, preds)

# Mean vectors
mu_g = np.mean(IQ_g, axis=0)
mu_e = np.mean(IQ_e, axis=0)

# Difference vector (signal direction)
d = mu_e - mu_g
d_unit = d / np.linalg.norm(d)

# Project all IQ points onto signal axis
proj_g = IQ_g @ d_unit
proj_e = IQ_e @ d_unit

# Compute noise variance (assume equal noise for both classes)
sigma2 = 0.5 * (np.var(proj_g) + np.var(proj_e))

# Compute SNR
snr = (np.linalg.norm(mu_e - mu_g) ** 2) / sigma2

figures = {"fidelity": fidelity, "SNR": snr}
with open("Rect_figures_ideal_qubit.json", "w") as f:
    json.dump(figures, f)

plt.figure(figsize=(6, 6))

plt.scatter(I_g, Q_g, color='blue', alpha=0.5, label='|g⟩ shots (Qubit Relaxation)')
plt.scatter(I_e, Q_e, color='red', alpha=0.5, label='|e⟩ shots (Qubit Relaxation)')
plt.plot(IQ_all, y_vals, 'k', label='Decision Boundary (Qubit Relaxation)')
plt.xlabel("I (integrated)")
plt.ylabel("Q (integrated)")
plt.title("IQ Distribution for Qubit States")
plt.legend()
plt.grid(True)
plt.ylim(min(Q_g + Q_e) - 5,max(Q_g + Q_e) + 5)
plt.xlim(min(I_g + I_e) - 5,max(I_g + I_e) + 5)
plt.savefig("Rect_IQ_3000_ideal_qubit.png")  # Save to file

plt.close()



