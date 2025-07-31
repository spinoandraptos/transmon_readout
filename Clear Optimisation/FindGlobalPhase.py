import re
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import correlate
from ReadoutSimulator import ReadoutSimulator
from scipy.optimize import differential_evolution

RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  

ref_e = np.load("Clear Optimisation/env_e_opt.npy")
ref_g = np.load("Clear Optimisation/env_g_opt.npy")

# ----------- PULSE PARAMS --------------------------

length = 1125.0
pad = 91.0e-9
ringdown1_amp = 0.004627128723203468
ringup1_amp = 0.648440470345643
ringdown1_time = 221.0e-9
ringup1_time = 350e-9 #300.0e-9
ringdown2_amp = -0.65
ringdown2_time = 300.0e-9
ringup2_amp = 0.0019805039111854927
ringup2_time = 299.0e-9
drive_amp = 0.0163084430717644
drive_time = 5.0e-9
threshold = 7.823733186454758e-07

# ----------- DO NOT MODIFY BELOW --------------------------
dt = 64e-9

# # Function to evaluate expressions with variables in YAML
# def evaluate_expression(expression, variables=None):
#     if variables:
#         for var, val in variables.items():
#             expression = re.sub(r'\b' + var + r'\b', str(val), expression)
#     try:
#         if not isinstance(expression, str):
#             expression = str(expression)
#         return eval(expression)
#     except (NameError, TypeError, SyntaxError) as e:
#         print(f"Error evaluating expression: {e} {expression}")
#         return None

# # Load the YAML file
# with open(f"{params_filepath}", "r") as file:
#     params = yaml.safe_load(file)

# if params is None:
#     raise ValueError("No parameters found in the YAML file.")

# # Extract parameters from YAML
# # chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi            # Dispersive shift
# # kappa = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate
# # sample_offset_ns = evaluate_expression(params["sample_offset_ns"]) * 1e-9

# def time_align(sim, ref):
#     corr = correlate(ref.real, sim.real, mode='full')
#     lag = np.argmax(corr) - (len(sim) - 1)
#     if lag > 0:
#         sim_aligned = np.pad(sim, (lag, 0), mode='constant')[:len(ref)]
#     elif lag < 0:
#         sim_aligned = np.pad(sim, (0, -lag), mode='constant')[-lag:len(ref)-lag]
#     else:
#         sim_aligned = sim
#     return sim_aligned


# def complex_corr(a, b):
#     return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

# def shape_mismatch(a, b):
#     return np.sum(np.abs(np.gradient(a.real) - np.gradient(b.real))**2)

# # def objective(params):
# #     global ref_e, ref_g

# #     phase, factor, sample_offset_ns, offset_r, offset_i = params
# #     sample_offset_ns = np.round(sample_offset_ns / 1e-9) * 1e-9  # nearest ns

# #     full_params = [
# #         kappa, chi, phase, sample_offset_ns, drive_amp,
# #         ringup1_time, ringdown1_time, drive_time, ringup2_time, ringdown2_time,
# #         ringup1_amp, ringdown1_amp, ringup2_amp, ringdown2_amp, pad, offset_r, offset_i
# #     ]

# #     RRSim = ReadoutSimulator(*full_params)
# #     env_e, env_g = RRSim.get_envelopes(factor)

# #     scale_e = np.linalg.norm(ref_e) / (np.linalg.norm(env_e) + 1e-12)
# #     scale_g = np.linalg.norm(ref_g) / (np.linalg.norm(env_g) + 1e-12)

# #     env_e_scaled = env_e * scale_e
# #     env_g_scaled = env_g * scale_g

# #     L = min(len(env_e_scaled), len(ref_e))
# #     env_e_scaled = env_e_scaled[:L]
# #     env_g_scaled = env_g_scaled[:L]
# #     ref_e_local = ref_e[:L]
# #     ref_g_local = ref_g[:L]

# #     phase_offset_e = np.angle(np.vdot(ref_e_local, env_e_scaled))
# #     phase_offset_g = np.angle(np.vdot(ref_g_local, env_g_scaled))

# #     env_e_scaled *= np.exp(1j * phase_offset_e)
# #     env_g_scaled *= np.exp(1j * phase_offset_g)

# #     corr_e = complex_corr(env_e_scaled, ref_e_local)
# #     corr_g = complex_corr(env_g_scaled, ref_g_local)

# #     nmse_e = np.sum(np.abs(env_e_scaled - ref_e_local)**2) / (np.sum(np.abs(ref_e_local)**2) + 1e-12)
# #     nmse_g = np.sum(np.abs(env_g_scaled - ref_g_local)**2) / (np.sum(np.abs(ref_g_local)**2) + 1e-12)

# #     diff_mag = np.sum((np.abs(env_e_scaled) - np.abs(ref_e_local))**2) + \
# #            np.sum((np.abs(env_g_scaled) - np.abs(ref_g_local))**2)

# #     diff = (nmse_e + nmse_g) +  (2 - corr_e - corr_g)  + diff_mag

# #     return diff

# def objective(params):
#     global ref_e, ref_g

#     phase, factor, sample_offset_ns, offset_r, offset_i, kappa, chi = params
#     sample_offset_ns = np.round(sample_offset_ns / 1e-9) * 1e-9  # nearest ns
#     kappa *=  2 * np.pi * 1e6
#     chi *= 2 * np.pi * 1e6

#     full_params = [
#         kappa, chi, phase, sample_offset_ns, drive_amp,
#         ringup1_time, ringdown1_time, drive_time, ringup2_time, ringdown2_time,
#         ringup1_amp, ringdown1_amp, ringup2_amp, ringdown2_amp, pad, offset_r, offset_i
#     ]

#     RRSim = ReadoutSimulator(*full_params)
#     env_e, env_g = RRSim.get_envelopes(factor)

#     scale_e = np.linalg.norm(ref_e) / (np.linalg.norm(env_e) + 1e-12)
#     scale_g = np.linalg.norm(ref_g) / (np.linalg.norm(env_g) + 1e-12)

#     env_e_scaled = env_e * scale_e
#     env_g_scaled = env_g * scale_g

#     L = min(len(env_e_scaled), len(ref_e))
#     env_e_scaled = env_e_scaled[:L]
#     env_g_scaled = env_g_scaled[:L]
#     ref_e_local = ref_e[:L]
#     ref_g_local = ref_g[:L]

#     phase_offset_e = np.angle(np.vdot(ref_e_local, env_e_scaled))
#     phase_offset_g = np.angle(np.vdot(ref_g_local, env_g_scaled))

#     env_e_scaled *= np.exp(1j * phase_offset_e)
#     env_g_scaled *= np.exp(1j * phase_offset_g)

#     corr_e = complex_corr(env_e_scaled, ref_e_local)
#     corr_g = complex_corr(env_g_scaled, ref_g_local)

#     nmse_e = np.sum(np.abs(env_e_scaled - ref_e_local)**2) / (np.sum(np.abs(ref_e_local)**2) + 1e-12)
#     nmse_g = np.sum(np.abs(env_g_scaled - ref_g_local)**2) / (np.sum(np.abs(ref_g_local)**2) + 1e-12)

#     diff_mag = np.sum((np.abs(env_e_scaled) - np.abs(ref_e_local))**2) + \
#            np.sum((np.abs(env_g_scaled) - np.abs(ref_g_local))**2)

#     diff = (nmse_e + nmse_g) +  (2 - corr_e - corr_g)  + diff_mag

#     return diff

# # bounds = [(0.0, 2.0), (-2.0, 2.0), (0e-9, 200e-9), (-10e-4, 10e-4), (-10e-4, 10e-4)]  # phase, factor, offset_ns, offset_r, offset_i
# # result = differential_evolution(
# #     objective,
# #     bounds,
# #     strategy='best1bin',
# #     maxiter=1000,
# #     popsize=30,
# #     tol=1e-5,
# #     mutation=(0.7, 1.2),
# #     recombination=0.6,
# # )

# # print(f"Optimal phase: {result.x[0]:.4f} π, Optimal Factor {result.x[1]:.4f}, Optimal offset {result.x[2]/1e-9:.1f} ns, Offset R: {result.x[3]:.4f}, Offset I: {result.x[4]:.4f}, Cost: {result.fun:.4f}")

# bounds = [(0.0, 2.0), (-2.0, 2.0), (0e-9, 200e-9), (-10e-4, 10e-4), (-10e-4, 10e-4), (0.0, 1.5), (0.0, 1.5)]  # phase, factor, offset_ns, offset_r, offset_i, kappa, chi
# result = differential_evolution(
#     objective,
#     bounds,
#     strategy='best1bin',
#     maxiter=1000,
#     popsize=30,
#     tol=1e-5,
#     mutation=(0.7, 1.2),
#     recombination=0.6,
# )

# print(f"Optimal phase:   {result.x[0]:.4f} π")
# print(f"Optimal factor:  {result.x[1]:.4f}")
# print(f"Optimal offset:  {result.x[2] / 1e-9:.1f} ns")
# print(f"Offset Re:       {result.x[3]/1e-4:.4f} e-4")
# print(f"Offset Im:       {result.x[4]/1e-4:.4f} e-4")
# print(f"kappa:           {result.x[5]:.4f}")
# print(f"chi:             {result.x[6]:.4f}")
# print(f"Final cost:      {result.fun:.4f}")

# optimal_phase = result.x[0] 
# optimal_factor = result.x[1]
# optimal_offset = np.round(result.x[2]/1e-9) * 1e-9 
# optimal_offset_r = result.x[3]
# optimal_offset_i = result.x[4]
# optimal_kappa = result.x[5] * 2 * np.pi * 1e6  # Convert to rad/s
# optimal_chi = result.x[6] * 2 * np.pi * 1e6  # Convert to


optimal_phase = 1 
optimal_factor = -0.7126
optimal_offset = 0.4*1e-9
optimal_offset_r = 0.0002
optimal_offset_i = 0.0000
optimal_kappa = 0.336 * 2 * np.pi * 1e6  # Convert to rad/s
optimal_chi = 1.4997 * 2 * np.pi * 1e6  # Convert to


full_params = [
    optimal_kappa, optimal_chi, optimal_phase, optimal_offset, drive_amp,
    ringup1_time, ringdown1_time, drive_time, ringup2_time, ringdown2_time,
    ringup1_amp, ringdown1_amp, ringup2_amp, ringdown2_amp, pad, optimal_offset_r, optimal_offset_i
]

# Run simulation for plotting
RRSim = ReadoutSimulator(*full_params)
env_e, env_g = RRSim.get_envelopes(optimal_factor)

# (Optional) apply same normalization as optimizer used
scale_e = np.linalg.norm(ref_e) / (np.linalg.norm(env_e) + 1e-12)
scale_g = np.linalg.norm(ref_g) / (np.linalg.norm(env_g) + 1e-12)
env_e_scaled = env_e * scale_e
env_g_scaled = env_g * scale_g

L = min(len(env_e_scaled), len(ref_e))
env_e_scaled = env_e_scaled[:L]
env_g_scaled = env_g_scaled[:L]
ref_e = ref_e[:L]
ref_g = ref_g[:L]

pulse, t_pulse = RRSim.get_pulse()

t = np.arange(len(ref_e)) * dt  # adjust dt

plt.figure(figsize=(12, 8))

# Plot pulse
plt.subplot(3, 1, 1)
plt.plot(t_pulse / 1e-9, np.real(pulse), label='Pulse Real', color='purple')
plt.plot(t_pulse / 1e-9, np.imag(pulse), label='Pulse Imag', color='magenta', linestyle='--')
plt.title('Input Pulse')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot excited state envelope
plt.subplot(3, 1, 2)
plt.plot(t / 1e-9, np.real(ref_e), label='ref_e Re', color='blue')
plt.plot(t / 1e-9, np.real(env_e_scaled), '--', label='sim_e Re', color='blue', alpha=0.7)
plt.plot(t / 1e-9, np.imag(ref_e), label='ref_e Im', color='orange')
plt.plot(t / 1e-9, np.imag(env_e_scaled), '--', label='sim_e Im', color='orange', alpha=0.7)
plt.title('Excited State Envelope (|e⟩)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot ground state envelope
plt.subplot(3, 1, 3)
plt.plot(t / 1e-9, np.real(ref_g), label='ref_g Re', color='green')
plt.plot(t / 1e-9, np.real(env_g_scaled), '--', label='sim_g Re', color='green', alpha=0.7)
plt.plot(t / 1e-9, np.imag(ref_g), label='ref_g Im', color='red')
plt.plot(t / 1e-9, np.imag(env_g_scaled), '--', label='sim_g Im', color='red', alpha=0.7)
plt.title('Ground State Envelope (|g⟩)')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()