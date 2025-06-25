# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import optimise_pulse, cross_check_with_square, plot_optimal_clear, convert_numpy_types
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, Tag

ruamel_yaml = YAML()
ruamel_yaml.indent(mapping=2, sequence=4, offset=2)

# Suppress all warnings
warnings.filterwarnings("ignore")

MIN_WAVEFORM_VOLTAGE: float = -0.5  # V
MAX_WAVEFORM_VOLTAGE: float = 0.5
IMPROVEMENT_FACTOR: float = 0.25  # Factor to improve the pulse shape 
QUBIT_IDX = 0
QUBIT_EF_IDX = 1
READOUT_IDX = 2
CAVITY_IDX = 3

# Define file paths
params_filepath = str(Path.cwd()) + "/SystemParam.yaml"
clear_filepath = str(Path.cwd()) + "/ClearParam.yaml"
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
with open(f"{params_filepath}", "r") as file:
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

# Ideally we do not want a readout exceeding 500ns
duration = 500e-9                                                      # Duration of the simulation       
dt = 1e-9                                                               # Sampling time step 
pulse_start = 50e-9
pulse_width = 300e-9

steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = np.inf, 0, np.inf, 0, None, None, None
steady_params, reset_params, threshold_steady, threshold_reset = None, None, 1e-5, 1e-8

while steady_time_clear > (1-IMPROVEMENT_FACTOR) * steady_time_square or reset_time_clear > (1-IMPROVEMENT_FACTOR) * reset_time_square:
    # Thresholds for determining if the cavity is steady or reset
    threshold_steady = 1e-5
    threshold_reset = 1e-8

    steady_params, reset_params, threshold_steady, threshold_reset = optimise_pulse(
        duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset
    ) 

    steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = cross_check_with_square(
        steady_params, reset_params, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset)

print(f"Steady state time CLEAR: {steady_time_clear/1e-9:.2f} ns")
print(f"Steady state time square: {steady_time_square/1e-9:.2f} ns")
print(f"Reset time CLEAR: {reset_time_clear/1e-9:.2f} ns")
print(f"Reset time square: {reset_time_square/1e-9:.2f} ns")

all_amps = [
        steady_params[2],  # ringup1_amp
        steady_params[3],  # ringdown1_amp
        reset_params[2],   # ringup2_amp
        reset_params[3],   # ringdown2_amp
    ]

max_amp = max(abs(a) for a in all_amps)
scale_factor = 0.5 / max_amp  # So that the largest amp becomes ±0.5

modes = None
with open("modes.yml") as modes_file:
    modes = ruamel_yaml.load(modes_file)

rr_readout_pulse = modes[READOUT_IDX]['operations']['rr_readout_pulse']

tag = Tag(handle='!', suffix='<ClearReadoutPulse>', handles={'!': '!'})
tag.select_transform(False)
rr_readout_pulse.yaml_set_ctag(tag)

# Add the rest
rr_readout_pulse.update({
    'has_optimized_weights': True,
    'length': 2000,
    'pad': 600,
    'threshold': 0.000624145668837496,
    'total_I_amp': 0.05,
    'total_length': 2600,
    'weights': r'C:\Users\qcrew\project-template\config\weights\20230721_145912_weights.npz',
    'ringdown1_amp': convert_numpy_types(steady_params[3] * scale_factor),
    'ringup1_amp': convert_numpy_types(steady_params[2] * scale_factor),
    'ringdown1_time': convert_numpy_types(round(steady_params[1]/4e-9)),
    'ringup1_time': convert_numpy_types(round(steady_params[0]/4e-9)),
    'ringdown2_amp': convert_numpy_types(reset_params[3] * scale_factor),
    'ringdown2_time': convert_numpy_types(round(reset_params[1]/4e-9)),
    'ringup2_amp': convert_numpy_types(reset_params[2] * scale_factor),
    'ringup2_time': convert_numpy_types(round(reset_params[0]/4e-9)),
    'drive_amp': convert_numpy_types(steady_params[4] * scale_factor),
    'drive_time': convert_numpy_types(round((pulse_width - steady_params[0] - steady_params[1])/4e-9))
})

with open(f"{clear_filepath}", 'w') as f:
    ruamel_yaml.dump(modes, f)

print(f"CLEAR pulse params saved to {clear_filepath}")

plot_optimal_clear(duration, dt, envelope, photon_0, photon_s)