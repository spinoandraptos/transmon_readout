import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import re
import yaml
from pathlib import Path

# ----------- TO MODIFY --------------------------

RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
use_CLEAR_pulse = True  #If false, uses square pulse instead of CLEAR puls

# ----------- PULSE PARAMS --------------------------

pad = 1219.0e-9
ringdown1_amp = 0.01020561438022858
ringup1_amp = 0.1296189632998254
ringdown1_time = 82.0e-9
ringup1_time = 92.0e-9
ringdown2_amp = -0.09449178942983193
ringdown2_time = 86.0e-9
ringup2_amp = 0.02590972115163823
ringup2_time = 95.0e-9
drive_amp = 0.0832
drive_time = 1626.0e-9



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

phases = np.arange(0, 2 * np.pi, np.pi / 8)

# Setup subplot grid
n_phases = len(phases)
n_rows, n_cols = 4, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
axes = axes.flatten()  # Make axes 1D iterable

# Track distinguishabilities
dist_list = []
best_idx = None
max_dist = -np.inf

for idx, phase in enumerate(phases):

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

    distinguishability = np.sum(np.abs(b_out_e_sampled - b_out_g_sampled)**2)
    dist_list.append(distinguishability)

    if distinguishability > max_dist:
        max_dist = distinguishability
        best_idx = idx

    ax = axes[idx]

    ax.plot(t_sampled * 1e9, np.real(b_out_g_sampled), label='R (|g⟩)', color='blue')
    ax.plot(t_sampled * 1e9, np.imag(b_out_g_sampled), label='I (|g⟩)', color='orange')
    ax.plot(t_sampled * 1e9, np.real(b_out_e_sampled), label='R (|e⟩)', color='blue', linestyle='--')
    ax.plot(t_sampled * 1e9, np.imag(b_out_e_sampled), label='I (|e⟩)', color='orange', linestyle='--')

    ax.set_title(f'Phase = {phase/np.pi:.2f} π')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Field")
    ax.grid(True)
    
    # --- Add distinguishability on plot ---
    ax.text(0.05, 0.95,
            f'Diff = {distinguishability:.4f}',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    if idx == 0:
        ax.legend()

# Hide unused axes (if any)
for j in range(idx + 1, len(axes)):
    if axes[j] in fig.axes:
        fig.delaxes(axes[j])

# Highlight the subplot with max distinguishability
dist_array = np.array(dist_list)
max_dist = np.max(dist_array)
tol = 1e-10  # tolerance for float comparison

# Step 2: Get all indices with max value
max_indices = np.where(np.abs(dist_array - max_dist) < tol)[0]

# Step 3: Highlight all subplots with max distinguishability
for i in max_indices:
    ax = axes[i]
    ax.set_title(ax.get_title() + " ⬅️ MAX", color='red')
    for spine in ax.spines.values():
        spine.set_color('red')
        spine.set_linewidth(2)

plt.suptitle(f"{RR} Return Signal Dynamics for Varying Phase Offsets", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.savefig(f"Clear Optimisation\{RR}_phase_sweep.png", dpi=300)
plt.close()
