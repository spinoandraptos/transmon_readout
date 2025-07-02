# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import optimise_pulse, cross_check_with_square, plot_optimal_clear, convert_numpy_types
from ruamel.yaml import YAML
from ruamel.yaml.comments import Tag
import qm as qm 
from qcore.helpers import Stage
from qcore.modes import *
from qcore.pulses import *
import numpy as np
from config.experiment_config import MODES_CONFIG
from pathlib import Path
from qm.octave import *

MIN_WAVEFORM_VOLTAGE: float = -0.5  # V
MAX_WAVEFORM_VOLTAGE: float = 0.5
IMPROVEMENT_FACTOR: float = 0.3  # Factor to improve the pulse shape 
MAX_TRIES = 1000
NORM_FACTOR = 0.5
CYCLE_LEN = 4
RRC_IDX = 0
QC_IDX = 1
DRIVE_C_IDX = 2
RRB_IDX = 3
QB_IDX = 4
QB_EF_IDX = 5
CAVB_IDX = 6
QA_IDX = 7
RRA_IDX = 8
CAVA_IDX = 9
DRIVE_A_IDX = 10
QA_EF_IDX = 11
QA_GF_IDX = 12
DIVISION_LEN = 16
DURATION = 30

# Modify these 
READOUT_IDX = RRA_IDX
RR = 'rrA' #rrA, rrC

improvement_factor = IMPROVEMENT_FACTOR
# ruamel_yaml = YAML()
# ruamel_yaml.indent(mapping=2, sequence=4, offset=2)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define file paths
params_filepath = str(Path.cwd()) + "\\scripts\\CLEAR\\SystemParam.yml"
clear_filepath = str(Path.cwd()) + "\\scripts\\CLEAR\\ClearParam.txt"
modes_filepath = str(Path.cwd()) + "\\config\\modes.yml"
env_filepath = str(Path.cwd()) + "\\scripts\\CLEAR\\optimal_clear_envelope.png"
photon_filepath = str(Path.cwd()) + "\\scripts\\CLEAR\\optimal_clear_photon.png"
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

num_tries = 0

while steady_time_clear > (1-improvement_factor) * steady_time_square or reset_time_clear > (1-improvement_factor) * reset_time_square:
    
    if num_tries == 1000:
        improvement_factor - 0.05
        num_tries = 0
        print(f"{MAX_TRIES} reached, decreasing improvement_factor to {improvement_factor}")

    # Thresholds for determining if the cavity is steady or reset
    threshold_steady = 1e-5
    threshold_reset = 1e-8

    steady_params, reset_params, threshold_steady, threshold_reset = optimise_pulse(
        duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset
    ) 

    steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = cross_check_with_square(
        steady_params, reset_params, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset)
    
    num_tries += 1

print(f"Optimisation completed with steady improvement of {1-steady_time_clear/steady_time_square}% and reset improvement of {1-reset_time_clear/reset_time_square}%")
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

def write_pulse_config(filename, config):
    with open(filename, 'w') as f:
        for key, value in config.items():
            if isinstance(value, str):
                if value.startswith("DigitalWaveform(") or value == "None":
                    f.write(f'                {key}={value},\n')
                else:
                    f.write(f'                {key}="{value}",\n')
            else:
                f.write(f'                {key} = {value},\n')

# modes = None
# with open(modes_filepath) as modes_file:
#     modes = ruamel_yaml.load(modes_file)


# rr_readout_pulse = modes[READOUT_IDX]['operations'][f'{RR}_CLEAR_readout_pulse']
# tag = Tag(handle='!', suffix='<ClearReadoutPulse>', handles={'!': '!'})
# tag.select_transform(False)
# rr_readout_pulse.yaml_set_ctag(tag)

ringdown1_time = convert_numpy_types(round(steady_params[1]/1e-9))
ringup1_time = convert_numpy_types(round(steady_params[0]/1e-9))
ringdown2_time = convert_numpy_types(round(reset_params[1]/1e-9))
ringup2_time = convert_numpy_types(round(reset_params[0]/1e-9))
drive_time = convert_numpy_types(round((pulse_width - steady_params[0] - steady_params[1])/1e-9))

ringdown1_amp = convert_numpy_types(steady_params[3] * scale_factor)
ringup1_amp = convert_numpy_types(steady_params[2] * scale_factor)
ringdown2_amp = convert_numpy_types(reset_params[3] * scale_factor)
ringup2_amp = convert_numpy_types(reset_params[2] * scale_factor)
drive_amp = convert_numpy_types(steady_params[4] * scale_factor)

length = (ringdown1_time + ringup1_time + ringdown2_time + ringup2_time + drive_time)
# padding = (CYCLE_LEN - length % CYCLE_LEN) if length % CYCLE_LEN else 0
padding = DURATION * (4 * DIVISION_LEN) - length
total_length = length + padding

pulse_config = {
    "name": f"{RR}_CLEAR_readout_pulse",
    'I_ampx': NORM_FACTOR,
    'Q_ampx': 0.0,
    'length': int(length),
    'pad': int(padding),
    'ringdown1_amp': ringdown1_amp,
    'ringup1_amp': ringup1_amp,
    'ringdown1_time': ringdown1_time, 
    'ringup1_time': ringup1_time,
    'ringdown2_amp': ringdown2_amp,
    'ringdown2_time': ringdown2_time,
    'ringup2_amp': ringup2_amp,
    'ringup2_time': ringup2_time,
    'drive_amp': drive_amp,
    'drive_time': drive_time,
    "digital_marker": "DigitalWaveform(\"ADC_ON\")",
}

# Write to file
write_pulse_config(clear_filepath, pulse_config)


# # Set pulse shape
# rr_readout_pulse.update({
#     'I_ampx': NORM_FACTOR,
#     'Q_ampx': 0.0,
#     'has_optimized_weights': False,
#     'length': int(length),
#     'pad': int(padding),
#     'total_length': total_length,
#     'ringdown1_amp': ringdown1_amp,
#     'ringup1_amp': ringup1_amp,
#     'ringdown1_time': ringdown1_time, 
#     'ringup1_time': ringup1_time,
#     'ringdown2_amp': ringdown2_amp,
#     'ringdown2_time': ringdown2_time,
#     'ringup2_amp': ringup2_amp,
#     'ringup2_time': ringup2_time,
#     'drive_amp': drive_amp,
#     'drive_time': drive_time
# })

# # Reset weights, threshold to null
# rr_readout_pulse.pop('weights', None)
# rr_readout_pulse.pop('threshold', None)

# with open(modes_filepath, 'w') as f:
#     ruamel_yaml.dump(modes, f)

print(f"CLEAR pulse params saved to {clear_filepath}")

plot_optimal_clear(duration, dt, envelope, photon_0, photon_s, env_filepath, photon_filepath)