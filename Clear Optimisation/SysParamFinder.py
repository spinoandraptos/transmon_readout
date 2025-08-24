import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ClearFormatter import ClearFormatter
from scipy.optimize import differential_evolution
from ReadoutSimulator import ReadoutSimulator, evaluate_expression

# ----------- Select right system params --------------------------
RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  

# Load reference envelope traces obtained from Train Weights
ref_e = np.load("Clear Optimisation/rr_simple_e.npy")
ref_g = np.load("Clear Optimisation/rr_simple_g.npy")

mode = 0  # 0 for CLEAR pulse, 1 for square pulse

# ----------- PULSE PARAMS --------------------------

clear = ClearFormatter(

                I_ampx = 0.2,
                Q_ampx = 0.0,
                length = 64*15, #1291, #1231,
                pad = 0, #53, #49,
                ringdown1_amp =1,
                ringup1_amp = 1,
                ringdown1_time = 64*4,
                ringup1_time = 64*4,
                ringdown2_amp = -0.85, #-0.3537721949299031,
                ringdown2_time = 64*2, #150, #90,
                ringup2_amp = 0,
                ringup2_time = 64*3, #182,
                drive_amp = -0.85,
                drive_time = 64*2,

)

# ----------- FITTING PARAMS ------------------------------

strategy = 'best1bin'
maxiter = 200
popsize = 30
tol = 1e-12
mutation = (0.7, 1.2)
recombination = 0.8
disp = True
workers = 10

# ----------- DO NOT MODIFY BELOW --------------------------
dt = 64e-9

# Load the YAML file
with open(f"{params_filepath}", "r") as file:
    params = yaml.safe_load(file)

if params is None:
    raise ValueError("No parameters found in the YAML file.")

# Extract parameters from YAML
chi = evaluate_expression(params["chi"])               # Dispersive shift
# kappa = evaluate_expression(params["kappa"]) * 2 * np.pi         # Resonator decay rate

def complex_corr(a, b):
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

def objective(params):
    global ref_e, ref_g

    # --- Extract params ---
    phase, factor, sample_offset_ns, offset_r, offset_i, kappa, ramp, attenuation = params
    sample_offset_ns = np.round(sample_offset_ns / 1e-9) * 1e-9  
    kappa *= 1e6
    ramp *= 1e-9

    # Run simulation
    full_params = [clear.ringup1_time, clear.ringdown1_time,clear.drive_time, clear.ringup2_time, clear.ringdown2_time,
                   clear.ringup1_amp, clear.ringdown1_amp, clear.ringup2_amp, clear.ringdown2_amp,
                   attenuation, kappa, ramp, chi, phase, 
                   sample_offset_ns, clear.drive_amp, offset_r, offset_i, clear.pad]

    RRSim = ReadoutSimulator(*full_params)
    env_g_scaled, env_e_scaled, _, _ = RRSim.get_envelopes(factor, mode)

    # Truncate to match reference length
    L = min(len(env_e_scaled), len(ref_e))
    env_e_scaled = env_e_scaled[:L]
    env_g_scaled = env_g_scaled[:L]
    ref_e_local = ref_e[:L]
    ref_g_local = ref_g[:L]

    # Compute NMSE (already normalized so mostly for small mismatches)
    nmse_e = np.mean(np.abs(env_e_scaled - ref_e_local)**2)
    nmse_g = np.mean(np.abs(env_g_scaled - ref_g_local)**2)

    cost = nmse_e + nmse_g

    return cost

bounds = [(0.0, 2.0), (0.0, 1.0), (0e-9, 500e-9), (-10e-4, 10e-4), (-10e-4, 10e-4), (0.0, 1.5), (0, 200), (0.0, 1.0)]  
            # phase, factor, offset_ns, offset_r, offset_i, kappa, ramp, attenuation
result = differential_evolution(
    objective,
    bounds,
    strategy=strategy,
    maxiter=maxiter,
    popsize=popsize,
    tol=tol,
    mutation=mutation,
    recombination=recombination,
    disp=disp,
    workers=workers
)

print("\n")
print(f"phase:              {result.x[0]:.2f}")
print(f"factor:             {result.x[1]:.2f}")
print(f"offset_ns:          {result.x[2] / 1e-9:.1f}")
print(f"offset_r:           {result.x[3]:.3f}e-4")
print(f"offset_i:           {result.x[4]:.3f}e-4")
print(f"kappa:          {result.x[5]:.3f}e6")
print(f"ramp:               {result.x[6]:.2f}e-9")
print(f"attenuation:        {result.x[7] / 1e-3:.1f}e-3")

optimal_phase = result.x[0] 
optimal_factor = result.x[1]
optimal_offset = np.round(result.x[2]/1e-9) * 1e-9 
optimal_offset_r = result.x[3]
optimal_offset_i = result.x[4]
optimal_kappa = result.x[5] * 1e6  
optimal_ramp = result.x[6] * 1e-9
optimal_attenuation = result.x[7]

optimal_chi = chi


full_params = [
    clear.ringup1_time, clear.ringdown1_time, clear.drive_time, clear.ringup2_time, clear.ringdown2_time,
    clear.ringup1_amp, clear.ringdown1_amp, clear.ringup2_amp, clear.ringdown2_amp, 
    optimal_attenuation, optimal_kappa, optimal_ramp, optimal_chi, optimal_phase, 
    optimal_offset, clear.drive_amp, optimal_offset_r, optimal_offset_i, clear.pad
]

# Run simulation for plotting
RRSim = ReadoutSimulator(*full_params)
env_g_scaled, env_e_scaled, _, _ = RRSim.get_envelopes(optimal_factor, mode)

L = min(len(env_e_scaled), len(ref_e))
env_e_scaled = env_e_scaled[:L]
env_g_scaled = env_g_scaled[:L]
ref_e = ref_e[:L]
ref_g = ref_g[:L]

pulse, t_pulse = RRSim.get_pulse()

t = np.arange(len(ref_e)) * dt  

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