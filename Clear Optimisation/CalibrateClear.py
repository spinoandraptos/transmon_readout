# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearOptimiser import  plot_optimal_clear, convert_numpy_types, optimise_pulse, cross_check_with_square, tune_drive_for_photon
import numpy as np
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# ------------------------ MODIFIABLE PARAMS --------------------------------------

NORM_FACTOR = 1.3 # I_ampx
MAX_TRIES = 15
CYCLE_LEN = 4
DIVISION_LEN = 16

RR = 'rrB' #rrA, rrC

drives = {
    'rrA':  0.2,
    'rrB':  0.2,
    'rrC':  0.175,
    'rr':   0.65
}

dt = 1e-9                                                              
pulse_start = 0e-9
buffer = 32e-9

drive_photons = 1

ringup1_range = (4e-9, 300e-9)      # Ringup1 time
ringdown1_range = (4e-9, 300e-9)    # Ringdown1 time
ringup2_range = (4e-9, 300e-9)      # Ringup2 time
ringdown2_range = (4e-9, 300e-9)    # Ringdown2 time
drive_range = (4e-9, 600e-9)        # Drive time

# --------- uncomment for Local ------------------
params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"                              
clear_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_ClearParam.txt"
env_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_envelope.png"
b_out_g_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_bout_g.png"
b_out_e_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_bout_e.png"
diff_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_diff.png"
a_c_g_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_a_g.png"
a_c_e_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_a_e.png"
a_s_g_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_const_a_g.png"
a_s_e_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_const_a_e.png"

# --------- uncomment for Octave ------------------
# params_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_SystemParam.yml"
# clear_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_ClearParam.txt"
# env_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_envelope.png"
# b_out_g_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_bout_g.png"
# b_out_e_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_bout_e.png"
# diff_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_diff.png"
# a_c_g_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_a_g.png"
# a_c_e_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_a_e.png"
# a_s_g_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_const_a_g.png"
# a_s_e_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_const_a_e.png"

# ------------------------DO NOT TOUCH THESE UNLESS NEEDED --------------------------------------

MAX_DRIVE = drives[RR]

if __name__ == "__main__":
    
    params = None
    num_tries = 0
    sep_s, sep_c, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, envelope, a_c_g, a_c_e, a_s_g, a_s_e = 0, -np.inf, None, None, None, None, None, None, None, None, None, None, None
    CLEAR_params = None
    

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
    phase = evaluate_expression(params["phase"]) * np.pi  # Complex phase of the input field 

    drive_amp = tune_drive_for_photon(drive_photons, chi, k)
    print(f"Drive amplitude to reach ⟨n⟩ ≈ {drive_photons}: {drive_amp:.4f}")

    # Memory to keep track of the best params thus far to guide search
    best_sep_so_far = 0
    best_ringup_params_so_far = []
    best_drive_time = []
    best_ringdown_params_so_far = []
    best_CLEAR_params_so_far = []

    # To introduce random jumps when stagnating towards local maximum
    stagnate_count = 0

    last_integral_clear = 0
    randomise = False

    while num_tries < MAX_TRIES:

        randomise = False

        if stagnate_count > 2:
            stagnate_count = 0
            randomise = True
            print("=== Improvement stagnating: Randomising ===")

        print(f"---------Try {num_tries}------------------")
        CLEAR_params = optimise_pulse(
            buffer, phase, chi, k, pulse_start, drive_amp, best_CLEAR_params_so_far, randomise, MAX_DRIVE, ringup1_range, ringdown1_range, ringup2_range, ringdown2_range, drive_range
        )
        
        sep_s, sep_c, _, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, envelope, a_c_g, a_c_e, a_s_g, a_s_e = cross_check_with_square(
            CLEAR_params, buffer, phase, chi, k, pulse_start, drive_amp
        )
        
        # print("=== Optimising Results ===")
        # print(f"square integral {sep_s}, clear integral {sep_c}")
        # print(f"Current improvement of {(sep_c/sep_s - 1)* 100:.2f}%")

        if sep_c > best_sep_so_far:
            best_sep_so_far = sep_c
            best_CLEAR_params_so_far = CLEAR_params

        last_integral_clear = sep_c

        if ((sep_c - last_integral_clear)/last_integral_clear) <= 0.05:
            stagnate_count += 1

        num_tries += 1

    # print(f"Optimisation completed with integral improvement of {(sep_c/sep_s - 1)* 100:.2f}%")
    # print(f"Integral CLEAR: {sep_c:.8f}")
    # print(f"Integral square: {sep_s:.8f}")
    print(f"Best integral CLEAR: {best_sep_so_far:.8f}")
    
    highest_drive = max(abs(best_CLEAR_params_so_far[2]), abs(best_CLEAR_params_so_far[3]), abs(best_CLEAR_params_so_far[6]), abs(best_CLEAR_params_so_far[7])) 

    scale_factor = MAX_DRIVE / highest_drive

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

    ringdown1_time = convert_numpy_types(round(best_CLEAR_params_so_far[1]/1e-9))
    ringup1_time = convert_numpy_types(round(best_CLEAR_params_so_far[0]/1e-9))
    ringdown2_time = convert_numpy_types(round(best_CLEAR_params_so_far[5]/1e-9))
    ringup2_time = convert_numpy_types(round(best_CLEAR_params_so_far[4]/1e-9))
    drive_time = convert_numpy_types(round(best_CLEAR_params_so_far[8]/1e-9))
    buffer = convert_numpy_types(round(buffer/1e-9))

    ringup1_amp = best_CLEAR_params_so_far[2] * scale_factor
    ringdown1_amp = best_CLEAR_params_so_far[3] * scale_factor
    drive_amp = drive_amp * scale_factor
    ringup2_amp = best_CLEAR_params_so_far[6] * scale_factor
    ringdown2_amp = best_CLEAR_params_so_far[7] * scale_factor

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

    best_CLEAR_params_so_far[2] = ringup1_amp
    best_CLEAR_params_so_far[3] = ringdown1_amp
    best_CLEAR_params_so_far[6] = ringup2_amp
    best_CLEAR_params_so_far[7] = ringdown2_amp

    sep_s, sep_c, t_eval, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, envelope, a_c_g, a_c_e, a_s_g, a_s_e = cross_check_with_square(
        best_CLEAR_params_so_far, buffer*1e-9, phase, chi, k, pulse_start, drive_amp
    )

    plot_optimal_clear(t_eval* 1e9, envelope, b_out_c_e, b_out_c_g, diff_c, diff_s, a_c_g, a_c_e, a_s_g, a_s_e, env_filepath, b_out_g_filepath, b_out_e_filepath, diff_filepath, a_c_g_filepath, a_c_e_filepath, a_s_g_filepath, a_s_e_filepath)
    print(f"CLEAR pulse figs saved")

