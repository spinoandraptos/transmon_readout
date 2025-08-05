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

full_params = [clear.ringup1_time, clear.ringdown1_time, clear.drive_time, clear.ringup2_time, clear.ringdown2_time,
                clear.ringup1_amp, clear.ringdown1_amp, clear.ringup2_amp, clear.ringdown2_amp,
                kappa_int, kappa_ext, ramp, chi, phase, 
                sample_offset_ns, clear.drive_amp, offset_r, offset_i, clear.pad]

RRSim = ReadoutSimulator(*full_params)
env_e, env_g = RRSim.get_envelopes(factor, mode)
t_sampled = np.arange(len(env_e)) * dt

# ----------------- UNCOMMENT IF NEEDED (Diff Envelope Plot) --------------------------------------------------

fig, ax1 = plt.subplots(figsize=(7,3))

if mode:
    label_prefix = "Square"
else:
    label_prefix = "CLEAR"

ax1.plot(t_sampled * 1e9, np.real(env_g), label=f'{label_prefix}  R (|g⟩)', color='blue')
ax1.plot(t_sampled * 1e9, np.imag(env_g), label=f'{label_prefix}  I (|g⟩)', color='orange')

ax1.plot(t_sampled * 1e9, np.real(env_e), label=f'{label_prefix}  R (|e⟩)', color='blue', linestyle='--')
ax1.plot(t_sampled * 1e9, np.imag(env_e), label=f'{label_prefix}  I (|e⟩)', color='orange', linestyle='--')


ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Field amplitude")
ax1.legend(loc='upper left')
ax1.grid(True)

plt.suptitle("Return Signal Dynamics (Sampled Every 64 ns)")
plt.tight_layout()
plt.show()


# ----------------- UNCOMMENT IF NEEDED (Diff Envelopes But on 2 Plots) --------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

# if mode:
#     label_prefix = "Square"
# else:
#     label_prefix = "CLEAR"

# # Plot for |g⟩
# ax1.plot(t_sampled * 1e9, np.real(env_g), label=f'{label_prefix} R (|g⟩)', color='blue')
# ax1.plot(t_sampled * 1e9, np.imag(env_g), label=f'{label_prefix} I (|g⟩)', color='orange')
# ax1.set_ylabel("Amplitude")
# ax1.legend(loc='upper right')
# ax1.set_title("Output Field for |g⟩")

# # Plot for |e⟩
# ax2.plot(t_sampled * 1e9, np.real(env_e), label=f'{label_prefix} R (|e⟩)', color='blue')
# ax2.plot(t_sampled * 1e9, np.imag(env_e), label=f'{label_prefix} I (|e⟩)', color='orange')
# ax2.set_xlabel("Time (ns)")
# ax2.set_ylabel("Amplitude")
# ax2.legend(loc='upper right')
# ax2.set_title("Output Field for |e⟩")

# # Final formatting
# plt.suptitle("Return Signal Dynamics (Sampled Every 64 ns)", y=1.02)
# plt.tight_layout()
# plt.grid(True)


# ----------------- UNCOMMENT IF NEEDED (Diff Envelope Integral Print) --------------------------------------------------

distinguishability = np.sum(np.abs(env_e - env_g)**2)
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

# ----------------- UNCOMMENT IF NEEDED (Print Pulse Shape) --------------------------------------------------
# fig, ax1 = plt.subplots(figsize=(7,3))

# ax1.plot(t_eval * 1e9, b_in_vals * np.exp(-1j * phase), color='blue')

# ax1.set_xlabel("Time (ns)")
# ax1.set_ylabel("Envelope")
# ax1.legend(loc='upper left')
# ax1.grid(True)

# plt.suptitle("Envelope of CLEAR Pulse")
# plt.tight_layout()
# plt.show()

# ----------------- UNCOMMENT IF NEEDED (Print Diff Envelope) --------------------------------------------------
diff = env_e - env_g
fig, ax1 = plt.subplots(figsize=(7,3))  
ax1.plot(t_sampled * 1e9, np.real(diff), label='Diff (R)', color='blue')
ax1.plot(t_sampled * 1e9, np.imag(diff), label='Diff (I)', color='orange')
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Envelope")
ax1.legend(loc='upper left')
ax1.grid(True)

plt.suptitle("Diff Pulse")
plt.tight_layout()
plt.show()
