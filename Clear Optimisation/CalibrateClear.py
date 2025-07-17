# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import  plot_optimal_clear, convert_numpy_types, optimise_pulse, cross_check_with_square, tune_drive_for_photon
import numpy as np
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# ------------------------ MODIFIABLE PARAMS --------------------------------------

NORM_FACTOR = 1.0 # I_ampx
MAX_TRIES = 1000
CYCLE_LEN = 4
DIVISION_LEN = 16
IMPROVEMENT_FACTOR = 0.5

RR = 'rr'

drives = {
    'rr':  0.65,
}

duration = 1500e-9                                                            
dt = 1e-9                                                              
pulse_start = 0e-9
pulse_width = 300e-9
buffer = 128e-9
phase = np.pi * 1.75

drive_photons = 1

num_tries = 0

# --------- uncomment for Local ------------------
# params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"                              
# clear_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_ClearParam.txt"
# env_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_envelope.png"
# b_out_g_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_bout_g.png"
# b_out_e_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_bout_e.png"
# diff_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_diff.png"


# --------- uncomment for Octave ------------------
params_filepath = str(Path.cwd()) + f"\\{RR}_SystemParam.yml"
clear_filepath = str(Path.cwd()) + f"\\{RR}_ClearParam.txt"
env_filepath = str(Path.cwd()) + f"\\{RR}_optimal_clear_envelope.png"
b_out_g_filepath = str(Path.cwd()) + f"\\{RR}_optimal_clear_bout_g.png"
b_out_e_filepath = str(Path.cwd()) + f"\\{RR}_optimal_clear_bout_e.png"
diff_filepath = str(Path.cwd()) + f"\\{RR}_optimal_clear_diff.png"

# ------------------------DO NOT TOUCH THESE UNLESS NEEDED --------------------------------------

MAX_DRIVE = drives[RR]
integral_s, integral_c, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, envelope = 0, -np.inf, None, None, None, None, None, None, None
CLEAR_params = None, None, None

if __name__ == "__main__":
    
    params = None
    improvement_factor = IMPROVEMENT_FACTOR

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
    chi = evaluate_expression(params["coupling"]["chi"]) * 2 * np.pi        # Dispersive shift, cross non-linearity
    k = evaluate_expression(params["resonator"]["kappa"]) * 2 * np.pi       # Resonator decay rate

    drive_amp = tune_drive_for_photon(drive_photons, chi, k)
    print(f"Drive amplitude to reach ⟨n⟩ ≈ {drive_photons}: {drive_amp:.4f}")

    # Memory to keep track of the best params thus far to guide search
    best_integral_so_far = 0
    best_ringup_params_so_far = []
    best_ringdown_params_so_far = []

    # To introduce random jumps when stagnating towards local maximum
    stagnate_count = 0

    last_integral_clear = 0
    randomise = False

    while integral_c < (1+improvement_factor) * integral_s:
        
        if num_tries == 100:
            improvement_factor - 0.1
            print(f"{MAX_TRIES} reached, decreasing improvement_factor to {improvement_factor}")
            num_tries = 0

        randomise = False

        if stagnate_count > 5:
            stagnate_count = 0
            randomise = True
            print("=== Improvement stagnating: Randomising ===")

        best_params_so_far = np.concatenate([best_ringup_params_so_far, best_ringdown_params_so_far])

        print(f"---------Try {num_tries}------------------")
        CLEAR_params = optimise_pulse(
            buffer, phase, chi, k, pulse_start, pulse_width, drive_amp, best_ringup_params_so_far, best_ringdown_params_so_far, randomise
        )
        
        integral_s, integral_c, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, envelope = cross_check_with_square(
            CLEAR_params, buffer, phase, chi, k, pulse_start, pulse_width, drive_amp
        )
        
        print("=== Optimising Results ===")
        print(f"square integral {integral_s}, clear integral {integral_c}")
        print(f"Current improvement of {(integral_c/integral_s - 1)* 100:.2f}%")

        if integral_c > best_integral_so_far:
            best_integral_so_far = integral_c
            best_ringup_params_so_far = [CLEAR_params[0], CLEAR_params[1], CLEAR_params[2], CLEAR_params[3]]
            best_ringdown_params_so_far = [CLEAR_params[4], CLEAR_params[5], CLEAR_params[6], CLEAR_params[7]]

        last_integral_clear = integral_c

        if ((integral_c - last_integral_clear)/last_integral_clear) <= 0.05:
            stagnate_count += 1

        num_tries += 1

    print(f"Optimisation completed with integral improvement of {(integral_c/integral_s - 1)* 100:.2f}%")
    print(f"Integral CLEAR: {integral_c:.8f}")
    print(f"Integral square: {integral_s:.8f}")

    all_amps = [
            CLEAR_params[2],    # ringup1_amp
            CLEAR_params[3],    # ringdown1_amp
            CLEAR_params[6],    # ringup2_amp
            CLEAR_params[7],    # ringdown2_amp
        ]
    
    scale_factor = MAX_DRIVE / abs(CLEAR_params[2])

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

    ringdown1_time = convert_numpy_types(round(CLEAR_params[1]/1e-9))
    ringup1_time = convert_numpy_types(round(CLEAR_params[0]/1e-9))
    ringdown2_time = convert_numpy_types(round(CLEAR_params[5]/1e-9))
    ringup2_time = convert_numpy_types(round(CLEAR_params[4]/1e-9))
    drive_time = convert_numpy_types(round(pulse_width/1e-9))
    buffer = convert_numpy_types(round(buffer/1e-9))

    ringup1_amp = CLEAR_params[2] * scale_factor
    ringdown1_amp = CLEAR_params[3] * scale_factor
    drive_amp = drive_amp * scale_factor
    ringup2_amp = CLEAR_params[6] * scale_factor
    ringdown2_amp = CLEAR_params[7] * scale_factor

    length = (ringdown1_time + ringup1_time + ringdown2_time + ringup2_time + drive_time)
    padding = buffer + ((4 * DIVISION_LEN) - (length + buffer) % (4 * DIVISION_LEN))
    total_length = length + padding

    pulse_config = {
        "name": f"{RR}_opt_CLEAR_readout_pulse",
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
    print(f"CLEAR pulse params saved to {clear_filepath}")

    t_drive = ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time
    t_total = t_drive + padding
    t_span = (0, t_total)
    t_eval = np.linspace(*t_span, 1000)

    plot_optimal_clear(t_eval, envelope, b_out_c_e, b_out_c_g, diff_c, diff_s, env_filepath, b_out_g_filepath, b_out_e_filepath, diff_filepath)
    print(f"CLEAR pulse figs saved")

