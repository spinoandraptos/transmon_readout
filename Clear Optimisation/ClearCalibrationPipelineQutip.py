# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import  plot_optimal_clear, convert_numpy_types, optimise_pulse_simul, cross_check_with_square_simul, tune_drive_for_photon
import numpy as np
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# ------------------------ MODIFIABLE PARAMS --------------------------------------

NORM_FACTOR = 1.0 #I_ampx
QUBIT_STATE = 0 # 0 or 1
IMPROVEMENT_FACTOR_STEADY: float = 0.7  # Factor to improve the pulse shape
IMPROVEMENT_FACTOR_RESET: float = 0.55  # Factor to improve the pulse shape 
TUNE_DRIVE_VERBOSE = False
CALIBRATE_CLEAR_VERBOSE = False

MAX_TRIES = 1000
CYCLE_LEN = 4
DIVISION_LEN = 16
DURATION = 50

RR = 'rr'

drives = {
    'rr':  0.65,
}

duration = 2000e-9                                                            
dt = 1e-9                                                              
pulse_start = 50e-9
pulse_width = 800e-9
drive_photons = 1
drive_upper_lim = 1e10
drive_lower_lim = 0

threshold_steady, threshold_reset = 1e-2, 1e-3
num_tries = 0

# --------- uncomment for Local ------------------
# params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"                              
# clear_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_ClearParam.txt"
# env_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_{QUBIT_STATE}_clear_envelope.png"
# photon_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_{QUBIT_STATE}_clear_photon.png"

# --------- uncomment for Octave ------------------
params_filepath = str(Path.cwd()) + f"\\{RR}_SystemParam.yml"
clear_filepath = str(Path.cwd()) + f"\\{RR}_{QUBIT_STATE}_ClearParam.txt"
env_filepath = str(Path.cwd()) + f"\\{RR}_optimal_{QUBIT_STATE}_clear_envelope.png"
photon_filepath = str(Path.cwd()) + f"\\{RR}_optimal_{QUBIT_STATE}_clear_photon.png"

# ------------------------DO NOT TOUCH THESE UNLESS NEEDED --------------------------------------

MAX_DRIVE = drives[RR]
steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = np.inf, 0, np.inf, 0, None, None, None
steady_params, reset_params, CLEAR_params = None, None, None

if __name__ == "__main__":

    improvement_factor_reset = IMPROVEMENT_FACTOR_RESET
    improvement_factor_steady = IMPROVEMENT_FACTOR_STEADY
    
    params = None

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
    T1 = evaluate_expression(params["qubit"]["T1"])                         # Qubit T1          
    T2 = evaluate_expression(params["qubit"]["T2"])                         # Qubit T2 

    drive_amp = tune_drive_for_photon(TUNE_DRIVE_VERBOSE, QUBIT_STATE, chi, k, T1, T2, drive_photons, drive_lower_lim, drive_upper_lim)
    print(f"Drive amplitude to reach ⟨n⟩ ≈ {drive_photons}: {drive_amp:.4f}")

    # Memory to keep track of the best params thus far to guide search
    best_steady_time_so_far = np.inf
    best_reset_time_so_far = np.inf
    best_ringup_params_so_far = []
    best_ringdown_params_so_far = []

    # To introduce random jumps when stagnating towards local maximum
    stagnate_count = 0
    stagnate_count_steady = 0
    stagnate_count_reset = 0
    last_steady_time_clear = np.inf
    last_reset_time_clear = np.inf
    randomise_steady = False
    randomise_reset = False

    while steady_time_clear > (1-improvement_factor_steady) * steady_time_square or reset_time_clear > (1-improvement_factor_reset) * reset_time_square:
        
        if num_tries == 100:
            if steady_time_clear > (1-improvement_factor_steady) * steady_time_square:
                improvement_factor_steady - 0.05
                print(f"{MAX_TRIES} reached, decreasing improvement_factor to {improvement_factor_steady}")
            if reset_time_clear > (1-improvement_factor_reset) * reset_time_square:
                improvement_factor_reset - 0.05
                print(f"{MAX_TRIES} reached, decreasing improvement_factor to {improvement_factor_reset}")
            num_tries = 0

        randomise_steady = False
        randomise_reset = False

        if stagnate_count_reset > 5:
            stagnate_count_reset = 0
            randomise_reset = True
            print("=== Improvement stagnating: Randomising Reset Params ===")

        if stagnate_count_steady > 5:
            stagnate_count_steady = 0
            randomise_steady = True
            print("=== Improvement stagnating: Randomising Steady Params ===")

        best_params_so_far = np.concatenate([best_ringup_params_so_far, best_ringdown_params_so_far])
        print(f"---------Try {num_tries}------------------")
        CLEAR_params, threshold_steady_final, threshold_reset_final = optimise_pulse_simul(
            CALIBRATE_CLEAR_VERBOSE, QUBIT_STATE, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady, threshold_reset, drive_amp, best_ringup_params_so_far, best_ringdown_params_so_far, randomise_steady, randomise_reset
        )
        
        steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = cross_check_with_square_simul(
            QUBIT_STATE, CLEAR_params, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady_final, threshold_reset_final, drive_amp)
        
        print("=== Optimising Results ===")
        print(f"square reset time {reset_time_square}, clear reset time {reset_time_clear}, square steady time {steady_time_square}, clear steady time {steady_time_clear}")
        print(f"Current steady improvement of {(1-steady_time_clear/steady_time_square)* 100:.2f}% and reset improvement of {(1-reset_time_clear/reset_time_square)* 100:.2f}%")

        if steady_time_clear < best_steady_time_so_far:
            best_steady_time_so_far = steady_time_clear
            best_ringup_params_so_far = [CLEAR_params[0], CLEAR_params[1], CLEAR_params[2], CLEAR_params[3]]

        if reset_time_clear < best_reset_time_so_far:
            best_reset_time_so_far = reset_time_clear
            best_ringdown_params_so_far = [CLEAR_params[4], CLEAR_params[5], CLEAR_params[6], CLEAR_params[7]]

        last_reset_time_clear = reset_time_clear
        last_steady_time_clear = steady_time_clear

        if ((last_reset_time_clear - reset_time_clear)/last_reset_time_clear) <= 0.05:
            stagnate_count_reset += 1

        if ((last_steady_time_clear - steady_time_clear)/last_steady_time_clear) <= 0.05:
            stagnate_count_steady += 1


        num_tries += 1

    print(f"Optimisation completed with steady improvement of {(1-steady_time_clear/steady_time_square)* 100:.2f}% and reset improvement of {(1-reset_time_clear/reset_time_square) * 100:.2f}%")
    print(f"Steady state time CLEAR: {steady_time_clear/1e-9:.2f} ns")
    print(f"Steady state time square: {steady_time_square/1e-9:.2f} ns")
    print(f"Reset time CLEAR: {reset_time_clear/1e-9:.2f} ns")
    print(f"Reset time square: {reset_time_square/1e-9:.2f} ns")

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
    drive_time = convert_numpy_types(round((pulse_width - CLEAR_params[0] - CLEAR_params[1])/1e-9))

    ringup1_amp = CLEAR_params[2] * scale_factor
    ringdown1_amp = CLEAR_params[3] * scale_factor
    drive_amp = drive_amp * scale_factor
    ringup2_amp = CLEAR_params[6] * scale_factor
    ringdown2_amp = CLEAR_params[7] * scale_factor

    length = (ringdown1_time + ringup1_time + ringdown2_time + ringup2_time + drive_time)
    padding = DURATION * (4 * DIVISION_LEN) - length
    total_length = length + padding

    pulse_config = {
        "name": f"{RR}_{QUBIT_STATE}_CLEAR_readout_pulse",
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

    plot_optimal_clear(duration, dt, envelope, photon_0, photon_s, env_filepath, photon_filepath)
    print(f"CLEAR pulse figs saved to {env_filepath} and {photon_filepath}")

