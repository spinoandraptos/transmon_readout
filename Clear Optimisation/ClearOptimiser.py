import re
import yaml
import optuna
import numpy as np
from pathlib import Path
from skopt.space import Real
from ReadoutSimulator import ReadoutSimulator, evaluate_expression

# ----------- TO MODIFY --------------------------

RR = 'rr'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
alpha_clear = 3e8
alpha_time = 1e4

# For Optuna optimisation
N_calls = 1000
N_jobs = 1
random_state = None

drives = {
    'rrA':  0.08,
    'rrB':  0.10,
    'rrC':  0.175,
    'rr':   0.20,
    'rrFullEnjoy': 0.08,
    'rrkyoto': 0.50,
    'rrbris': 0.50,
}

# ----------- PULSE PARAMS --------------------------
max_drive = drives[RR]
drive_amp = max_drive

space = [
    Real(4e-9, 100e-9, name='ringup1_time'),
    Real(4e-9, 200e-9, name='ringdown1_time'),
    Real(4e-9, 700e-9, name='drive_time'),
    Real(4e-9, 200e-9, name='ringup2_time'),
    Real(4e-9, 100e-9, name='ringdown2_time'),
    Real(drive_amp*1.5, drive_amp * 2, name='ringup1_amp'),
    Real(0, drive_amp, name='ringdown1_amp'),
    Real(0, drive_amp, name='ringup2_amp'),
    Real(-drive_amp * 2, -drive_amp, name='ringdown2_amp'),
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
kappa_int = evaluate_expression(params["kappa_int"])                                    # Internal resonator kappa
kappa_ext = evaluate_expression(params["kappa_ext"])                                    # External resonator kappa
factor = evaluate_expression(params["factor"])                                          # Detuning factor (0-1)
offset_r = evaluate_expression(params["offset_re"])                                     # Offset for real simulation envelope origin
offset_i = evaluate_expression(params["offset_im"])                                     # Offset for imaginary simulation envelope origin
ramp = evaluate_expression(params["ramp"])                                              # Smoothing ramp of pulse (nonlinear effects)

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
    sys_params = [kappa_int, kappa_ext, ramp, chi, phase, sample_offset_ns, drive_amp, offset_r, offset_i]
    full_params = list(CLEAR_params) + sys_params
    RRSim = ReadoutSimulator(*full_params)
    cost = RRSim.cost(alpha_clear, alpha_time)
    return cost

def objective(trial):
    params = []
    for dim in space:
        low, high, name = dim.low, dim.high, dim.name
        # Use log sampling if the original Real had prior='log-uniform'
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

ringup1_time_ns    = round(p['ringup1_time'] / 1e-9)
ringdown1_time_ns  = round(p['ringdown1_time'] / 1e-9)
drive_time_ns      = round(p['drive_time'] / 1e-9)
ringup2_time_ns    = round(p['ringup2_time'] / 1e-9)
ringdown2_time_ns  = round(p['ringdown2_time'] / 1e-9)

length = ringup1_time_ns + ringdown1_time_ns + drive_time_ns + ringup2_time_ns + ringdown2_time_ns
pad = (4 * DIVISION_LEN) - length % (4 * DIVISION_LEN)

sys_params = [kappa_int, kappa_ext, ramp, chi, phase, sample_offset_ns, drive_amp, offset_r, offset_i]

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

optuna.visualization.plot_optimization_history(study).show()
