import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ClearFormatter import ClearFormatter
from ReadoutSimulator import ReadoutSimulator, evaluate_expression

# ----------- TO MODIFY --------------------------

RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
mode = 0  #If 1, uses square pulse instead of CLEAR pulse

# ----------- PULSE PARAMS --------------------------

clear = ClearFormatter(

    length = 1061,
    pad = 27,
    ringdown1_amp = 0.030845450564527973,
    ringup1_amp = 0.3997222782200416,
    ringdown1_time = 139,
    ringup1_time = 37,
    ringdown2_amp = -0.21516629230666512,
    ringdown2_time = 86,
    ringup2_amp = 0.010860770682225731,
    ringup2_time = 196,
    drive_amp = 0.2,
    drive_time = 603

)

# ----------- DO NOT MODIFY BELOW --------------------------
dt = 64e-9

# Load the YAML file
with open(f"{params_filepath}", "r") as file:
    params = yaml.safe_load(file)

if params is None:
    raise ValueError("No parameters found in the YAML file.")

# Extract parameters from YAML
chi = evaluate_expression(params["chi"])                                                # Dispersive shift, cross non-linearity
phase = evaluate_expression(params["phase"])                                            # Global phase of the input field 
sample_offset_ns = np.round(evaluate_expression(params["offset_ns"])) * 1e-9            # Offset for start of sampling (TOF factor)
kappa_int = evaluate_expression(params["kappa_int"])                                    # Internal resonator kappa
kappa_ext = evaluate_expression(params["kappa_ext"])                                    # External resonator kappa
factor = evaluate_expression(params["factor"])                                          # Detuning factor (0-1)
offset_r = evaluate_expression(params["offset_re"])                                     # Offset for real simulation envelope origin
offset_i = evaluate_expression(params["offset_im"])                                     # Offset for imaginary simulation envelope origin
ramp = evaluate_expression(params["ramp"])                                              # Smoothing ramp of pulse (nonlinear effects)

phases = np.arange(0, 2 * np.pi, np.pi / 16)

# Setup subplot grid
n_phases = len(phases)
n_rows, n_cols = 8, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
axes = axes.flatten()  # Make axes 1D iterable

# Track distinguishabilities
dist_list = []
best_idx = None
max_dist = -np.inf

for idx, phase in enumerate(phases):

    full_params = [clear.ringup1_time, clear.ringdown1_time, clear.drive_time, clear.ringup2_time, clear.ringdown2_time,
                clear.ringup1_amp, clear.ringdown1_amp, clear.ringup2_amp, clear.ringdown2_amp,
                kappa_int, kappa_ext, ramp, chi, phase, 
                sample_offset_ns, clear.drive_amp, offset_r, offset_i, clear.pad]

    RRSim = ReadoutSimulator(*full_params)
    env_e, env_g = RRSim.get_envelopes(factor, mode)
    t_sampled = np.arange(len(env_e)) * dt

    # ----------------- UNCOMMENT IF NEEDED (Diff Envelope Plot) --------------------------------------------------

    distinguishability = np.sum(np.abs(env_e - env_g)**2)
    dist_list.append(distinguishability)

    if distinguishability > max_dist:
        max_dist = distinguishability
        best_idx = idx

    ax = axes[idx]

    if mode:
        label_prefix = "Square"
    else:
        label_prefix = "CLEAR"

    ax.plot(t_sampled * 1e9, np.real(env_g), label=f'{label_prefix}  R (|g⟩)', color='blue')
    ax.plot(t_sampled * 1e9, np.imag(env_g), label=f'{label_prefix}  I (|g⟩)', color='orange')
    ax.plot(t_sampled * 1e9, np.real(env_e), label=f'{label_prefix}  R (|e⟩)', color='blue', linestyle='--')
    ax.plot(t_sampled * 1e9, np.imag(env_e), label=f'{label_prefix}  I (|e⟩)', color='orange', linestyle='--')

    ax.set_title(f'Phase = {phase/np.pi:.2f} π')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Field")
    ax.grid(True)
    
    # --- Add distinguishability on plot ---
    ax.text(0.05, 0.95,
            f'Diff = {distinguishability:.6f}',
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
plt.savefig(f"Clear Optimisation/{RR}_phase_sweep.png", dpi=300)
plt.close()
