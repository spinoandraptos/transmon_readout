import yaml
import optuna
import numpy as np
from pathlib import Path
from skopt.space import Real
from ReadoutSimulator import ReadoutSimulator, evaluate_expression

# ----------- TO MODIFY --------------------------

RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  

alpha_sep = 2e2
alpha_clear = 4e16
alpha_time = 4e7
S_min = 1.0  # Minimum separation

# For Optuna optimisation
N_calls = 1000
N_jobs = 1
random_state = None
view_opt_history = False

drives = {
    'rrA':  0.08,
    'rrB':  0.10,
    'rrC':  0.175,
    'rr':   0.2,
    'rrFullEnjoy': 0.08,
    'rrkyoto': 0.50,
    'rrbris': 0.50,
}

# ----------- PULSE PARAMS --------------------------
max_drive = drives[RR]
drive_amp = max_drive

space = [
    Real(80e-9, 800e-9, name='ringup1_time'),
    Real(4e-9, 800e-9, name='ringdown1_time'),
    Real(4e-9, 800e-9, name='drive_time'),
    Real(4e-9, 800e-9, name='ringup2_time'),
    Real(80e-9, 800e-9, name='ringdown2_time'),
    Real(drive_amp , drive_amp * 5, name='ringup1_amp'),
    Real(0, drive_amp, name='ringdown1_amp'),
    Real(0, drive_amp, name='ringup2_amp'),
    Real(-drive_amp * 5, 0, name='ringdown2_amp'),
]

# ----------- DO NOT MODIFY BELOW --------------------------
DIVISION_LEN = 16
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
kappa = evaluate_expression(params["kappa"])                                            # Resonator kappa
factor = evaluate_expression(params["factor"])                                          # Detuning factor (0-1)
offset_r = evaluate_expression(params["offset_r"])                                      # Offset for real simulation envelope origin
offset_i = evaluate_expression(params["offset_i"])                                      # Offset for imaginary simulation envelope origin
ramp = evaluate_expression(params["ramp"])                                              # Smoothing ramp of pulse (nonlinear effects)
attenuation = evaluate_expression(params["attenuation"])                                # Attenuation of transmission line

'''
CLEAR Params in the order of
1) ringup1_time
2) ringdown1_time
3) drive_time
4) ringup2_time
5) ringdown2_time
6) ringup1_amp
7) ringdown1_amp
8) ringup2_amp
9) ringdown2_amp

sys Params in the order of 
1) kappa_int
2) kappa_ext
3) gain
4) chi
5) phase
6) sample_offset_ns
7) drive_amp
8) offset_r
9) offset_i
10) pad

'''

def cost_function(params):
    CLEAR_params = np.array(params)
    sys_params = [attenuation, kappa, ramp, chi, phase, sample_offset_ns, drive_amp, offset_r, offset_i]
    full_params = list(CLEAR_params) + sys_params
    RRSim = ReadoutSimulator(*full_params)
    cost = RRSim.cost(alpha_sep, alpha_clear, alpha_time, factor, S_min)
    return cost

def objective(trial):
    params = []
    for dim in space:
        low, high, name = dim.low, dim.high, dim.name
        
        if 'time' in name:
            # Convert low/high from seconds to integer steps
            low_steps = int(np.round(low / 1e-9))
            high_steps = int(np.round(high / 1e-9))
            steps = trial.suggest_int(name, low_steps, high_steps)
            val = steps * 1e-9  # convert back to seconds
        else:
            # Keep amplitudes as floats
            if getattr(dim, 'prior', None) == 'log-uniform':
                val = trial.suggest_float(name, low, high, log=True)
            else:
                val = trial.suggest_float(name, low, high)
        
        params.append(val)

    cost = cost_function(params)
    return cost

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=random_state)  # TPE is default
)

study.optimize(objective, n_trials=N_calls, n_jobs=N_jobs)
print(f"\nMinimum cost: {study.best_value:.4f}")

p = study.best_params

ringup1_time_ns    = round(p['ringup1_time'])
ringdown1_time_ns  = round(p['ringdown1_time'])
drive_time_ns      = round(p['drive_time'])
ringup2_time_ns    = round(p['ringup2_time'])
ringdown2_time_ns  = round(p['ringdown2_time'])

length = ringup1_time_ns + ringdown1_time_ns + drive_time_ns + ringup2_time_ns + ringdown2_time_ns
pad = (4 * DIVISION_LEN) - length % (4 * DIVISION_LEN)

sys_params = [attenuation, kappa, ramp, chi, phase, sample_offset_ns, drive_amp, offset_r, offset_i]

formatted = {
    'I_ampx' : 1.0,
    'Q_ampx' : 0.0,
    'length': length,
    'pad': pad,
    'ringdown1_amp': p['ringdown1_amp'],
    'ringup1_amp': p['ringup1_amp'],
    'ringdown1_time': ringdown1_time_ns,
    'ringup1_time': ringup1_time_ns,
    'ringdown2_amp': p['ringdown2_amp'],
    'ringdown2_time': ringdown2_time_ns,
    'ringup2_amp': p['ringup2_amp'],
    'ringup2_time': ringup2_time_ns,
    'drive_amp': sys_params[6],
    'drive_time': drive_time_ns,
}

# Print in requested format
print("\n🧾 Best Result (formatted):\n")
for k, v in formatted.items():
    print(f"    {k} = {v},")
print("\n")

# Save to file
with open(f"Clear Optimisation/{RR}_ClearParam.txt", "w") as f:
    for k, v in formatted.items():
        f.write(f"    {k} = {v},\n")

if view_opt_history:
    optuna.visualization.plot_optimization_history(study).show()
