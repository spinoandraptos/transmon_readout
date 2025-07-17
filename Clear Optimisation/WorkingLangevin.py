import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------- TO MODIFY --------------------------

kappa = 2 * np.pi * 0.170e6        # cavity decay rate
chi = 2 * np.pi * 1.05e6        # dispersive shift

ringdown1_amp = 0.2034264717601678 
ringup1_amp = 0.8 
ringdown1_time = 50.0e-9
ringup1_time = 80.0e-9
ringdown2_amp = -0.4473838685161403 
ringdown2_time = 60.0e-9
ringup2_amp = 0.1115762247846788 
ringup2_time = 45.0e-9
drive_amp = 0.3047675988177768 
drive_time = 1500.0e-9

buffer = 128e-9

phase = np.pi * 1.75 # Complex phase of the input field 

# ----------- DO NOT MODIFY --------------------------
pulse_start = 0.0
pulse_width = drive_time
t_drive = ringup1_time + ringdown1_time + drive_time + ringdown2_time + ringup2_time
t_total = t_drive + buffer

t_span = (0, t_total)
t_eval = np.linspace(*t_span, 1000)

def clear_pulse(t):
    if t <= pulse_start:
        return 0.0
    elif t <= pulse_start + ringup1_time:
        return ringup1_amp * np.exp(1j * phase)  # optionally phase
    elif t <= pulse_start + ringup1_time + ringdown1_time:
        return ringdown1_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width:
        return drive_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width + ringdown2_time:
        return ringdown2_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time:
        return ringup2_amp * np.exp(1j * phase)
    else:
        return 0.0

def square_pulse(t):
    return drive_amp * np.exp(1j * phase) if 0 < t < t_drive else 0.0

def cavity_dynamics(t, y, drive_fn, delta):
    alpha = y[0] + 1j * y[1]
    d_alpha = -(1j * delta + kappa/2) * alpha - np.sqrt(kappa) * drive_fn(t)
    return [d_alpha.real, d_alpha.imag]

def solve_for_state(delta):
    sol_clear = solve_ivp(cavity_dynamics, t_span, [0, 0], args=(clear_pulse, delta), t_eval=t_eval)
    sol_square = solve_ivp(cavity_dynamics, t_span, [0, 0], args=(square_pulse, delta), t_eval=t_eval)
    alpha_clear = sol_clear.y[0] + 1j * sol_clear.y[1]
    alpha_square = sol_square.y[0] + 1j * sol_square.y[1]
    return alpha_clear, alpha_square

sol_clear_g, sol_square_g = solve_for_state(delta=0)
sol_clear_e, sol_square_e = solve_for_state(delta=+chi)

# Compute drive pulse values for each time point
b_in_vals = np.array([clear_pulse(t) for t in t_eval])
b_out_g = b_in_vals + np.sqrt(kappa) * sol_clear_g
b_out_e = b_in_vals + np.sqrt(kappa) * sol_clear_e
diff = b_out_e - b_out_g

n_clear_g = np.abs(sol_clear_g)**2
n_square_g = np.abs(sol_square_g)**2
n_clear_e = np.abs(sol_clear_e)**2
n_square_e = np.abs(sol_square_e)**2

# ----------------- UNCOMMENT --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7, 10))

ax1.plot(t_eval * 1e9, n_clear_g, label='CLEAR (|g⟩)', color='blue')
ax1.plot(t_eval * 1e9, n_square_g, label='Square (|g⟩)', color='blue', linestyle='--')
ax2.plot(t_eval * 1e9, n_clear_e, label='CLEAR (|e⟩)', color='red')
ax2.plot(t_eval * 1e9, n_square_e, label='Square (|e⟩)', color='red', linestyle='--')

ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Photon Number")
ax1.legend(loc='upper left')
ax1.grid(True)

ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Photon Number")
ax2.legend(loc='upper left')
ax2.grid(True)

plt.title("Photon Number & Drive Pulse")

plt.tight_layout()
plt.show()

# ----------------- UNCOMMENT --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7, 10))

# ax1.plot(t_eval * 1e9, np.real(sol_clear_g), label='CLEAR R (|g⟩)', color='blue')
# ax1.plot(t_eval * 1e9, np.imag(sol_clear_g), label='CLEAR I (|g⟩)', color='blue', linestyle='--')
# ax2.plot(t_eval * 1e9, np.real(sol_clear_e), label='CLEAR R (|e⟩)', color='red')
# ax2.plot(t_eval * 1e9, np.imag(sol_clear_e), label='CLEAR I (|e⟩)', color='red', linestyle='--')

ax1.plot(t_eval * 1e9, np.real(b_out_g), label='CLEAR R (|g⟩)', color='blue')
ax1.plot(t_eval * 1e9, np.imag(b_out_g), label='CLEAR I (|g⟩)', color='blue', linestyle='--')
ax2.plot(t_eval * 1e9, np.real(b_out_e), label='CLEAR R (|e⟩)', color='red')
ax2.plot(t_eval * 1e9, np.imag(b_out_e), label='CLEAR I (|e⟩)', color='red', linestyle='--')

ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Field amplitude")
ax1.legend(loc='upper left')
ax1.grid(True)

ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Field amplitude")
ax2.legend(loc='upper left')
ax2.grid(True)

plt.title("Return Signal Dynamics & Drive Pulse")
plt.tight_layout()
plt.show()

# ----------------- UNCOMMENT --------------------------------------------------
plt.plot(t_eval * 1e9, diff, label='Env Diff)', color='blue')
plt.grid(True)
plt.title("Envelope difference")

dt = t_eval[1] - t_eval[0]
distinguishability = np.sum(np.abs(diff)**2) * dt
print(f"Diff Integral: {distinguishability:.8f}")

plt.tight_layout()
plt.show()



# def steady_state(epsilon, delta, kappa, t):
#     amp = (2 * np.sqrt(kappa) * epsilon) / (1j * 2 * delta - kappa)
#     steady_state = amp * (1 - np.exp(-(kappa/2 + delta) * t))
#     return steady_state

# steady_state_g_vals = np.array([steady_state(drive_amp, -chi, kappa, t) for t in t_eval])
# steady_state_e_vals = np.array([steady_state(drive_amp, +chi, kappa, t) for t in t_eval])