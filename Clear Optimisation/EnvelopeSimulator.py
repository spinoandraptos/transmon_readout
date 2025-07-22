import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import re
import yaml
from pathlib import Path

# ----------- TO MODIFY --------------------------

RR = 'rrB'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
use_CLEAR_pulse = True  #If false, uses square pulse instead of CLEAR pulse

# ----------- PULSE PARAMS --------------------------

I_ampx = 1.0
Q_ampx = 0.0
length = 1504.0
pad = 96.0e-9
ringdown1_amp = 0.004701537537115074
ringup1_amp = 0.09497105824548026
ringdown1_time = 268.0e-9
ringup1_time = 296.0e-9
ringdown2_amp = -0.1
ringdown2_time = 79.0e-9
ringup2_amp = 0.09309044322892385
ringup2_time = 262.0e-9
drive_amp = 0.09403075073797584
drive_time = 599.0e-9

# ----------- DO NOT MODIFY BELOW --------------------------
def round_to_4(x):
    return 4e-9 * round(x / 4e-9)

ringup1_time = round_to_4(ringup1_time)
ringdown1_time = round_to_4(ringdown1_time)
ringup2_time = round_to_4(ringup2_time)
ringdown2_time = round_to_4(ringdown2_time)
drive_time = round_to_4(drive_time)

buffer = pad
pulse_start = 0.0
t_drive = ringup1_time + ringdown1_time + drive_time + ringdown2_time + ringup2_time
t_total = t_drive + buffer
dt = 1e-9 

# Sampling interval
sample_interval_ns = 64  # in ns
sample_offset_ns = 0           # Start sampling with an offset of 32 ns

t_eval = np.arange(0, t_total, dt)  # ensure inclusive endpoint
t_span = (t_eval[0], t_eval[-1])        # make sure solve_ivp agrees

sample_interval_steps = int(sample_interval_ns * 1e-9 / dt)
sample_offset_steps = int(sample_offset_ns * 1e-9 / dt)

# Function to evaluate expressions with variables in YAML
def evaluate_expression(expression, variables=None):
    if variables:
        for var, val in variables.items():
            expression = re.sub(r'\b' + var + r'\b', str(val), expression)
    try:
        if not isinstance(expression, str):
            expression = str(expression)
        return eval(expression)
    except (NameError, TypeError, SyntaxError) as e:
        print(f"Error evaluating expression: {e} {expression}")
        return None


# Load the YAML file
with open(f"{params_filepath}", "r") as file:
    params = yaml.safe_load(file)

if params is None:
    raise ValueError("No parameters found in the YAML file.")

# Extract parameters from YAML
chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi        # Dispersive shift, cross non-linearity
kappa = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate
phase = evaluate_expression(params["phase"]) * np.pi  # Complex phase of the input field 

def clear_pulse(t):
    if t <= pulse_start:
        return 0.0
    elif t <= pulse_start + ringup1_time:
        return ringup1_amp * np.exp(1j * phase)  # optionally phase
    elif t <= pulse_start + ringup1_time + ringdown1_time:
        return ringdown1_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + drive_time:
        return drive_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + drive_time + ringdown2_time:
        return ringdown2_amp * np.exp(1j * phase)
    elif t <= pulse_start + ringup1_time + ringdown1_time + drive_time + ringdown2_time + ringup2_time:
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
    if use_CLEAR_pulse:
        sol_clear = solve_ivp(cavity_dynamics, t_span, [0, 0], args=(clear_pulse, delta), t_eval=t_eval)
        alpha_clear = sol_clear.y[0] + 1j * sol_clear.y[1]
        return alpha_clear, None
    else:
        sol_square = solve_ivp(cavity_dynamics, t_span, [0, 0], args=(square_pulse, delta), t_eval=t_eval)
        alpha_square = sol_square.y[0] + 1j * sol_square.y[1]
        return None, alpha_square

sol_clear_g, sol_square_g = solve_for_state(delta=0)
sol_clear_e, sol_square_e = solve_for_state(delta=+chi)

if use_CLEAR_pulse:
    b_in_vals = np.array([clear_pulse(t) for t in t_eval])
    b_out_g = b_in_vals + np.sqrt(kappa) * sol_clear_g
    b_out_e = b_in_vals + np.sqrt(kappa) * sol_clear_e

else:
    b_in_vals = np.array([square_pulse(t) for t in t_eval])
    b_out_g = b_in_vals + np.sqrt(kappa) * sol_square_g
    b_out_e = b_in_vals + np.sqrt(kappa) * sol_square_e

# Sampled time and b_out values
sample_indices = np.arange(sample_offset_steps, len(t_eval), sample_interval_steps)

# Sample arrays
t_sampled = t_eval[sample_indices]
b_out_g_sampled = b_out_g[sample_indices]
b_out_e_sampled = b_out_e[sample_indices]

# ----------------- UNCOMMENT IF NEEDED (Diff Envelope Plot) --------------------------------------------------

fig, ax1 = plt.subplots(figsize=(7,3))

if use_CLEAR_pulse:

    ax1.plot(t_sampled * 1e9, np.real(b_out_g_sampled), label='CLEAR R (|g⟩)', color='blue')
    ax1.plot(t_sampled * 1e9, np.imag(b_out_g_sampled), label='CLEAR I (|g⟩)', color='orange')

    ax1.plot(t_sampled * 1e9, np.real(b_out_e_sampled), label='CLEAR R (|e⟩)', color='blue', linestyle='--')
    ax1.plot(t_sampled * 1e9, np.imag(b_out_e_sampled), label='CLEAR I (|e⟩)', color='orange', linestyle='--')

else:
    ax1.plot(t_sampled * 1e9, np.real(b_out_g_sampled), label='Square R (|g⟩)', color='blue')
    ax1.plot(t_sampled * 1e9, np.imag(b_out_g_sampled), label='Square I (|g⟩)', color='orange')

    ax1.plot(t_sampled * 1e9, np.real(b_out_e_sampled), label='Square R (|e⟩)', color='blue', linestyle='--')
    ax1.plot(t_sampled * 1e9, np.imag(b_out_e_sampled), label='Square I (|e⟩)', color='orange', linestyle='--')


ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Field amplitude")
ax1.legend(loc='upper left')
ax1.grid(True)

plt.suptitle("Return Signal Dynamics (Sampled Every 64 ns)")
plt.tight_layout()
plt.show()

# ----------------- UNCOMMENT IF NEEDED (Diff Envelope Integral Print) --------------------------------------------------

distinguishability = np.sum(np.abs(b_out_e_sampled - b_out_g_sampled)**2)
print(f"Diff Integral: {distinguishability:.8f}")

# ----------------- UNCOMMENT IF NEEDED (Resonator Photon Number) --------------------------------------------------

# n_clear_g = np.abs(sol_clear_g)**2
# n_square_g = np.abs(sol_square_g)**2
# n_clear_e = np.abs(sol_clear_e)**2
# n_square_e = np.abs(sol_square_e)**2

# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7, 10))

# ax1.plot(t_eval * 1e9, n_clear_g, label='CLEAR (|g⟩)', color='blue')
# ax1.plot(t_eval * 1e9, n_square_g, label='Square (|g⟩)', color='blue', linestyle='--')
# ax2.plot(t_eval * 1e9, n_clear_e, label='CLEAR (|e⟩)', color='red')
# ax2.plot(t_eval * 1e9, n_square_e, label='Square (|e⟩)', color='red', linestyle='--')

# ax1.set_xlabel("Time (ns)")
# ax1.set_ylabel("Photon Number")
# ax1.legend(loc='upper left')
# ax1.grid(True)

# ax2.set_xlabel("Time (ns)")
# ax2.set_ylabel("Photon Number")
# ax2.legend(loc='upper left')
# ax2.grid(True)

# plt.title("Photon Number & Drive Pulse")

# plt.tight_layout()
# plt.show()

# ----------------- UNCOMMENT IF NEEDED (Unsampled Resonator and Return Fields) --------------------------------------------------
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7, 10))

# # ax1.plot(t_eval * 1e9, np.real(sol_clear_g), label='CLEAR R (|g⟩)', color='blue')
# # ax1.plot(t_eval * 1e9, np.imag(sol_clear_g), label='CLEAR I (|g⟩)', color='blue', linestyle='--')
# # ax2.plot(t_eval * 1e9, np.real(sol_clear_e), label='CLEAR R (|e⟩)', color='red')
# # ax2.plot(t_eval * 1e9, np.imag(sol_clear_e), label='CLEAR I (|e⟩)', color='red', linestyle='--')

# ax1.plot(t_eval * 1e9, np.real(b_out_g), label='CLEAR R (|g⟩)', color='blue')
# ax1.plot(t_eval * 1e9, np.imag(b_out_g), label='CLEAR I (|g⟩)', color='blue', linestyle='--')
# ax2.plot(t_eval * 1e9, np.real(b_out_e), label='CLEAR R (|e⟩)', color='red')
# ax2.plot(t_eval * 1e9, np.imag(b_out_e), label='CLEAR I (|e⟩)', color='red', linestyle='--')

# ax1.set_xlabel("Time (ns)")
# ax1.set_ylabel("Field amplitude")
# ax1.legend(loc='upper left')
# ax1.grid(True)

# ax2.set_xlabel("Time (ns)")
# ax2.set_ylabel("Field amplitude")
# ax2.legend(loc='upper left')
# ax2.grid(True)

# plt.title("Return Signal Dynamics & Drive Pulse")
# plt.tight_layout()
# plt.show()
