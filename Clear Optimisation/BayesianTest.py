import re
import yaml
import numpy as np
from pathlib import Path
from skopt.space import Real
from skopt import gp_minimize
import matplotlib.pyplot as plt
from skopt.callbacks import VerboseCallback
from ReadoutSimulator import ReadoutSimulator
from skopt.plots import plot_convergence, plot_objective

# ----------- TO MODIFY --------------------------

RR = 'rrFullEnjoy'  
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"  
alpha_clear = 1e5
alpha_time = 1e3

# For bayesian optimisation
N_calls = 200
N_random_starts = 100
random_state = None
N_restarts_optimizer = 20
acq_func = 'EI'
noise_std_est = 0.01

drives = {
    'rrA':  0.08,
    'rrB':  0.10,
    'rrC':  0.175,
    'rr':   0.65,
    'rrFullEnjoy': 0.08,
}

# ----------- PULSE PARAMS --------------------------
max_drive = drives[RR]
drive_amp = max_drive

# ringup1_length_range = (4e-9, 600e-9)                               # Ringup1 length bounds
# ringdown1_length_range = (4e-9, 600e-9)                             # Ringdown1 length bounds
# ringup2_length_range = (4e-9, 600e-9)                               # Ringup2 length bounds
# ringdown2_length_range = (4e-9, 600e-9)                             # Ringdown2 length bounds
# drive_length_range = (4e-9, 600e-9)                                 # Drive length bounds

# ringup1_amp_factor = (drive_amp * 1/40, drive_amp * 40)             # Ringup1 amp bounds (taken as a factor of drive amp)
# ringdown1_amp_factor  = (drive_amp * 1/40, drive_amp * 40)          # Ringdown1 amp bounds (taken as a factor of drive amp)
# ringup2_amp_factor  = (drive_amp * 1/40, drive_amp * 40)            # Ringup2 amp bounds (taken as a factor of drive amp)
# ringdown2_amp_factor  = (-drive_amp * 40, -drive_amp * 1/40)        # Ringdown2 amp bounds (taken as a factor of drive amp)

space = [
    Real(4e-9, 300e-9, name='ringup1_time'),
    Real(4e-9, 300e-9, name='ringdown1_time'),
    Real(4e-9, 300e-9, name='drive_time'),
    Real(4e-9, 300e-9, name='ringup2_time'),
    Real(4e-9, 300e-9, name='ringdown2_time'),
    Real(drive_amp * 1/40, drive_amp * 40, name='ringup1_amp'),
    Real(drive_amp * 1/40, drive_amp * 40, name='ringdown1_amp'),
    Real(drive_amp * 1/40, drive_amp * 40, name='ringup2_amp'),
    Real(-drive_amp * 40, -drive_amp * 1/40, name='ringdown2_amp'),
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

def scale_clear_params(CLEAR_params, sys_params, MAX_DRIVE, use_dict):

    if use_dict:
        highest_drive = max(
            abs(CLEAR_params['ringup1_amp']),
            abs(CLEAR_params['ringdown1_amp']),
            sys_params[3],  # assuming this is correct
            abs(CLEAR_params['ringup2_amp']),
            abs(CLEAR_params['ringdown2_amp'])
        )
    else:
        highest_drive = max(abs(CLEAR_params[5]), abs(CLEAR_params[6]), sys_params[3], abs(CLEAR_params[7]), abs(CLEAR_params[8])) 

    scale_factor = MAX_DRIVE / highest_drive
    
    if use_dict:
        CLEAR_params['ringup1_amp'] *= scale_factor
        CLEAR_params['ringdown1_amp'] *= scale_factor
        CLEAR_params['ringup2_amp'] *= scale_factor
        CLEAR_params['ringdown2_amp'] *= scale_factor
    else:
        CLEAR_params[5] *= scale_factor
        CLEAR_params[6] *= scale_factor
        CLEAR_params[7] *= scale_factor
        CLEAR_params[8] *= scale_factor

    sys_params[3] *= scale_factor

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
    sys_params = [kappa, chi, phase, drive_amp]
    scale_clear_params(CLEAR_params, sys_params, max_drive, 0)
    full_params = sys_params + list(CLEAR_params)
    RRSim = ReadoutSimulator(*full_params)
    cost = RRSim.cost(alpha_clear, alpha_time)

    return cost

# --- 3. Run Bayesian optimization
result = gp_minimize(
    func=cost_function,                         # function to minimize
    dimensions=space,                           # parameter space
    n_calls=N_calls,                            # number of evaluations
    n_random_starts=N_random_starts,            # initial random samples
    random_state=random_state,                  # reproducibility
    acq_func=acq_func,
    noise=noise_std_est,
    n_restarts_optimizer=N_restarts_optimizer,  # Try 5–20
    callback=[VerboseCallback(n_total=N_calls)]
)

# --- 4. Show results
print("Best parameters found:")
for name, val in zip([s.name for s in space], result.x):
    if 'time' in name:
        # Convert seconds to nanoseconds
        val_ns = val / 1e-9
        print(f"{name}: {val_ns:.2f} ns")
    else:
        print(f"{name}: {val:.4f}")
print(f"\nMinimum cost: {result.fun:.4f}")

# Get best parameters
best_params = dict(zip([s.name for s in space], result.x))

ringup1_time_ns    = round(best_params['ringup1_time'] / 1e-9)
ringdown1_time_ns  = round(best_params['ringdown1_time'] / 1e-9)
drive_time_ns      = round(best_params['drive_time'] / 1e-9)
ringup2_time_ns    = round(best_params['ringup2_time'] / 1e-9)
ringdown2_time_ns  = round(best_params['ringdown2_time'] / 1e-9)

length = ringup1_time_ns + ringdown1_time_ns + drive_time_ns + ringup2_time_ns + ringdown2_time_ns
pad = (4 * DIVISION_LEN) - length % (4 * DIVISION_LEN)
sys_params = [kappa, chi, phase, drive_amp]

scale_clear_params(best_params, sys_params, max_drive, 1)

formatted = {
    'length': length,
    'pad': pad,
    'ringdown1_amp': best_params['ringdown1_amp'],
    'ringup1_amp': best_params['ringup1_amp'],
    'ringdown1_time': ringdown1_time_ns,
    'ringup1_time': ringup1_time_ns,
    'ringdown2_amp': best_params['ringdown2_amp'],
    'ringdown2_time': ringdown2_time_ns,
    'ringup2_amp': best_params['ringup2_amp'],
    'ringup2_time': ringup2_time_ns,
    'drive_amp': sys_params[3],  # fixed
    'drive_time': drive_time_ns,
}

# Print in requested format
print("\n🧾 Best Result (formatted):\n")
for k, v in formatted.items():
    print(f"    {k} = {v},")
print("\n")

# --- 5. Plot convergence
ax1 = plot_convergence(result)
ax1.set_title("Bayesian Optimization Convergence")

# --- Plot objective ---
fig_or_ax = plot_objective(result)  # Don't pass `plot_type`, it’s not a valid kwarg

# Handle title safely depending on return type
if hasattr(fig_or_ax, "suptitle"):  # Figure
    fig_or_ax.suptitle("Parameter Effects on Objective")
elif hasattr(fig_or_ax, "set_title"):  # Axes
    fig_or_ax.set_title("Parameter Effects on Objective")

plt.show()