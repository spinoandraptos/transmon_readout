# Read in system parameters
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import optimise_pulse, plot_optimal_clear, convert_numpy_types

# Suppress all warnings
warnings.filterwarnings("ignore")

MIN_WAVEFORM_VOLTAGE: float = -0.5  # V
MAX_WAVEFORM_VOLTAGE: float = 0.5

def min_max_scale(value: float, min_value=MIN_WAVEFORM_VOLTAGE, max_value=MAX_WAVEFORM_VOLTAGE) -> float:
    if value < min_value or value > max_value:
        raise ValueError(f"Value {value} is out of bounds [{min_value}, {max_value}]")
    return (value - min_value) / (max_value - min_value)

# Define file paths
params_filepath = "systemParam"
clear_filepath = "clearParam"
params = None

# Function to evaluate expressions with variables in YAML
def evaluate_expression(expression, variables=None):
    if variables:
        for var, val in variables.items():
            expression = re.sub(r'\b' + var + r'\b', str(val), expression)
    try:
      return eval(expression)
    except (NameError, TypeError, SyntaxError) as e:
      print(f"Error evaluating expression: {e}")
      return None


# Load the YAML file
with open(f"{params_filepath}.yaml", "r") as file:
    params = yaml.safe_load(file)

if params is None:
    raise ValueError("No parameters found in the YAML file.")

# Extract parameters from YAML
wr_lo = evaluate_expression(params["resonator"]["lo"]) * 2 * np.pi      # Resonator LO
wr_if = evaluate_expression(params["resonator"]["if"]) * 2 * np.pi      # Resonator IF  
wq_if =  evaluate_expression(params["qubit"]["if"]) * 2 * np.pi         # Qubit IF
wq_lo = evaluate_expression(params["qubit"]["lo"]) * 2 * np.pi          # Qubit LO
chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi        # Dispersive shift, cross non-linearity
k = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate
T1 = evaluate_expression(params["qubit"]["T1"])                         # Qubit T1          
T2 = evaluate_expression(params["qubit"]["T2"])                         # Qubit T2 

wr = wr_if + wr_lo                                                      # Resonator frequency
wq = wq_if + wq_lo                                                      # Qubit frequency
delta = abs(wr - wq)                                                    # Detuning between qubit and resonator

# Ideally we do not want a readout exceeding 1000ns
duration = 1000e-9                                                      # Duration of the simulation       
dt = 1e-9                                                               # Sampling time step 
pulse_start = 50e-9
pulse_width = 500e-9

steady_params, reset_params = optimise_pulse(
    duration, dt, chi, k, pulse_start, pulse_width
) 

all_amps = [
    steady_params[2],  # ringup1_amp
    steady_params[3],  # ringdown1_amp
    reset_params[2],   # ringup2_amp
    reset_params[3],   # ringdown2_amp
]

max_amp = max(abs(a) for a in all_amps)
scale_factor = 0.5 / max_amp  # So that the largest amp becomes ±0.5

steady_state_params = {
    'drive_amp': convert_numpy_types(scale_factor * steady_params[4]),  # also scaled!
}

population_params = {
    'ringup1_time': convert_numpy_types(steady_params[0]),
    'ringdown1_time': convert_numpy_types(steady_params[1]),
    'ringup1_amp': convert_numpy_types(steady_params[2] * scale_factor),
    'ringdown1_amp': convert_numpy_types(steady_params[3] * scale_factor),
}

clear_params = {
    'ringup2_time': convert_numpy_types(reset_params[0]),
    'ringdown2_time': convert_numpy_types(reset_params[1]),
    'ringup2_amp': convert_numpy_types(reset_params[2] * scale_factor),
    'ringdown2_amp': convert_numpy_types(reset_params[3] * scale_factor),
}

data = {
    "population": population_params,
    "steady-state": steady_state_params,
    "reset": clear_params
}

with open(f"{clear_filepath}.yaml", 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"CLEAR pulse params saved to {clear_filepath}.yaml")

plot_optimal_clear(
    steady_params, reset_params, duration, dt, chi, k, pulse_start, pulse_width)

