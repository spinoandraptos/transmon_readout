import re
import yaml
import optuna
import numpy as np
from pathlib import Path
from skopt.space import Real
from ReadoutSimulator import ReadoutSimulator

# ----------- TO MODIFY --------------------------

RR = 'rrB'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
alpha_clear = 1e6
alpha_time = 1e4

# For Optuna optimisation
N_calls = 2000
N_jobs = 4
random_state = None

drives = {
    'rrA':  0.08,
    'rrB':  0.15,
    'rrC':  0.175,
    'rr':   0.65,
    'rrFullEnjoy': 0.08,
    'rrkyoto': 0.50,
    'rrbris': 0.50,
}

# ----------- PULSE PARAMS --------------------------
max_drive = drives[RR]
drive_amp = max_drive

space = [
    Real(4e-9, 200e-9, name='ringup1_time'),
    Real(4e-9, 200e-9, name='ringdown1_time'),
    Real(4e-9, 200e-9, name='drive_time'),
    Real(4e-9, 200e-9, name='ringup2_time'),
    Real(4e-9, 200e-9, name='ringdown2_time'),
    Real(drive_amp * 1.01, drive_amp * 5, name='ringup1_amp'),
    Real(drive_amp * 1/5, drive_amp * 5, name='ringdown1_amp'),
    Real(drive_amp * 1/5, drive_amp * 5, name='ringup2_amp'),
    Real(-drive_amp * 10, -drive_amp * 1/10, name='ringdown2_amp'),
]

# ----------- DO NOT MODIFY BELOW --------------------------
DIVISION_LEN = 16

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
chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi            # Dispersive shift
kappa = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate
phase = evaluate_expression(params["phase"]) * np.pi                        # Global phase of the input field 
sample_offset_ns = evaluate_expression(params["sample_offset_ns"]) * 1e-9
                                       
def scale_clear_params(CLEAR_params, sys_params, drive_amp, use_dict):
    # Assume drive_amp is the target max amplitude
    if use_dict:
        # Find the largest absolute amp among the 4
        current_max = max(
            abs(CLEAR_params['ringup1_amp']),
            abs(CLEAR_params['ringdown1_amp']),
            abs(CLEAR_params['ringup2_amp']),
            abs(CLEAR_params['ringdown2_amp']),
        )
    else:
        current_max = max(
            abs(CLEAR_params[5]),
            abs(CLEAR_params[6]),
            abs(CLEAR_params[7]),
            abs(CLEAR_params[8]),
        )

    # Avoid divide-by-zero
    if current_max == 0:
        scale_factor = 1.0
    else:
        scale_factor = drive_amp / current_max

    if use_dict:
        CLEAR_params['ringup1_amp']     *= scale_factor
        CLEAR_params['ringdown1_amp']   *= scale_factor
        CLEAR_params['ringup2_amp']     *= scale_factor
        CLEAR_params['ringdown2_amp']   *= scale_factor
    else:
        CLEAR_params[5] *= scale_factor
        CLEAR_params[6] *= scale_factor
        CLEAR_params[7] *= scale_factor
        CLEAR_params[8] *= scale_factor

    sys_params[3] = drive_amp  # ensure the drive_amp in sys_params matches

'''
Params in the order of
1) ringup1_time
2) ringdown1_time
3) drive_time
4) ringup2_time
5) ringdown2_time
6) ringup1_amp
7) ringdown1_amp
8) ringup2_amp
9) ringdown2_amp
'''

def cost_function(params):
    CLEAR_params = np.array(params)
    sys_params = [kappa, chi, phase, sample_offset_ns, drive_amp]
    scale_clear_params(CLEAR_params, sys_params, max_drive, 0)
    full_params = sys_params + list(CLEAR_params)
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

print("Best parameters found:")
for key, val in study.best_params.items():
    if 'time' in key:
        print(f"{key}: {val / 1e-9:.2f} ns")
    else:
        print(f"{key}: {val:.4f}")
print(f"\nMinimum cost: {study.best_value:.4f}")

p = study.best_params

ringup1_time_ns    = round(p['ringup1_time'] / 1e-9)
ringdown1_time_ns  = round(p['ringdown1_time'] / 1e-9)
drive_time_ns      = round(p['drive_time'] / 1e-9)
ringup2_time_ns    = round(p['ringup2_time'] / 1e-9)
ringdown2_time_ns  = round(p['ringdown2_time'] / 1e-9)

length = ringup1_time_ns + ringdown1_time_ns + drive_time_ns + ringup2_time_ns + ringdown2_time_ns
pad = (4 * DIVISION_LEN) - length % (4 * DIVISION_LEN)

sys_params = [kappa, chi, phase, drive_amp]
scale_clear_params(p, sys_params, max_drive, use_dict=1)

formatted = {
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
    'drive_amp': sys_params[3],
    'drive_time': drive_time_ns,
}

# Print in requested format
print("\n🧾 Best Result (formatted):\n")
for k, v in formatted.items():
    print(f"    {k} = {v},")
print("\n")

optuna.visualization.plot_optimization_history(study).show()
