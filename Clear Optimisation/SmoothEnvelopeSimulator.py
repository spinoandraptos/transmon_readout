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

length = 1017.0
pad = 207.0e-9
ringdown1_amp = 0.09858331201112192
ringup1_amp = 0.09737521112845736
ringdown1_time = 295.0e-9
ringup1_time = 200.0e-9
ringdown2_amp = -0.07671705677027515
ringdown2_time = 228.0e-9
ringup2_amp = 0.1
ringup2_time = 257.0e-9
drive_amp = 0.024072565872138404
drive_time = 37.0e-9

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
sample_interval_ns = 64   # in ns
sample_offset_ns = 200      # Start sampling with an offset of 32 ns

t_eval = np.arange(0, t_total, dt) 
t_span = (t_eval[0], t_eval[-1])      

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
chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi            # Dispersive shift, cross non-linearity
kappa = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate
phase = evaluate_expression(params["phase"]) * np.pi                        # Complex phase of the input drive 

def smooth_cosine_edge(t, t0, duration, amp, phase):
    """ Smooth cosine-squared ramp between 0 and amp over `duration` starting at `t0`. """
    if t < t0 or t > t0 + duration:
        return 0.0
    tau = (t - t0) / duration
    window = np.sin(np.pi * tau / 2)**2  # cosine-squared ramp
    return amp * window * np.exp(1j * phase)

def clear_pulse(t):
    try:
        t0 = pulse_start
        t1 = t0 + ringup1_time
        t2 = t1 + ringdown1_time
        t3 = t2 + drive_time
        t4 = t3 + ringdown2_time
        t5 = t4 + ringup2_time

        if t < t0:
            return 0.0
        elif t <= t1:
            return smooth_cosine_edge(t, t0, ringup1_time, ringup1_amp, phase)
        elif t <= t2:
            return smooth_cosine_edge(t, t1, ringdown1_time, ringdown1_amp, phase)
        elif t <= t3:
            return drive_amp * np.exp(1j * phase)
        elif t <= t4:
            return smooth_cosine_edge(t, t3, ringdown2_time, ringdown2_amp, phase)
        elif t <= t5:
            return smooth_cosine_edge(t, t4, ringup2_time, ringup2_amp, phase)
        else:
            return 0.0

    except Exception as e:
        print(f"b_in error at t={t}: {e}")
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


# ----------------- UNCOMMENT IF NEEDED (Drive Envelope) --------------------------------------------------

# pulse_values = np.array([clear_pulse(t) for t in t_sampled])
# plt.figure(figsize=(10, 5))
# plt.plot(t_sampled * 1e9, np.real(pulse_values), label='Re[pulse]')
# plt.plot(t_sampled * 1e9, np.imag(pulse_values), label='Im[pulse]', linestyle='--')
# plt.title("CLEAR Pulse with Smoothed Edges")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# ----------------- UNCOMMENT IF NEEDED (Envelope Difference) --------------------------------------------------

diff = b_out_e_sampled - b_out_g_sampled
plt.figure(figsize=(10, 5))
plt.plot(t_sampled * 1e9, np.real(diff), label='Re[Diff]', color = 'blue')
plt.plot(t_sampled * 1e9, np.imag(diff), label='Im[Diff]', color = 'orange')
plt.title("CLEAR Pulse with Smoothed Edges")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
