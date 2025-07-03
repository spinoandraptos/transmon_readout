# Read in system parameters
from pathlib import Path
import yaml
import re
import numpy as np
import warnings
from ClearSimulationHelper import optimise_pulse, cross_check_with_square, plot_optimal_clear, convert_numpy_types, optimise_pulse_simul, cross_check_with_square_simul
import numpy as np
from pathlib import Path

MIN_WAVEFORM_VOLTAGE: float = -0.5  # V
MAX_WAVEFORM_VOLTAGE: float = 0.5
IMPROVEMENT_FACTOR: float = 0.3  # Factor to improve the pulse shape 
MAX_TRIES = 1000
NORM_FACTOR = 0.1 #I_ampx
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
DURATION = 35

# Modify these 
READOUT_IDX = RRA_IDX
RR = 'rrC' #rrA, rrC

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    improvement_factor = IMPROVEMENT_FACTOR

    # Define file paths
    # --------- uncomment for Local ------------------
    params_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_SystemParam.yml"
    clear_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_ClearParam.txt"
    env_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_envelope.png"
    photon_filepath = str(Path.cwd()) + f"/Clear Optimisation/{RR}_optimal_clear_photon.png"

    # --------- uncomment for Octave ------------------
    # params_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_SystemParam.yml"
    # clear_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_ClearParam.txt"
    # env_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_envelope.png"
    # photon_filepath = str(Path.cwd()) + f"\\scripts\\CLEAR\\{RR}_optimal_clear_photon.png"
    
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
    duration = 3000e-9                                                      # Duration of the simulation       
    dt = 1e-9                                                               # Sampling time step 
    pulse_start = 50e-9
    pulse_width = 1500e-9

    steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = np.inf, 0, np.inf, 0, None, None, None
    steady_params, reset_params, CLEAR_params, threshold_steady, threshold_reset = None, None, None, 1e-2, 1e-4

    num_tries = 0


    while steady_time_clear > (1-improvement_factor) * steady_time_square or reset_time_clear > (1-improvement_factor) * reset_time_square:
        
        if num_tries == 250:
            improvement_factor - 0.05
            num_tries = 0
            print(f"{MAX_TRIES} reached, decreasing improvement_factor to {improvement_factor}")

        # ------- UNCOMMENT FOR SEPARATE OPT --------------
        # steady_params, reset_params, threshold_steady_final, threshold_reset_final = optimise_pulse(
        #     duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset
        # ) 

        # ------- UNCOMMENT FOR SIMULTANEOUS OPT --------------
        CLEAR_params, threshold_steady_final, threshold_reset_final = optimise_pulse_simul(
            duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset
        )

        # ------- UNCOMMENT FOR SEPARATE OPT --------------
        # steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = cross_check_with_square(
        #     steady_params, reset_params, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady_final, threshold_reset_final)

        # ------- UNCOMMENT FOR SIMULTANEOUS OPT --------------      
        steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s = cross_check_with_square_simul(
            CLEAR_params, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady_final, threshold_reset_final)
        
        print("=== Optimising Results ===")
        # print(f"square reset time {reset_time_square}, clear reset time {reset_time_clear}, square steady time {steady_time_square}, clear steady time {steady_time_clear}")
        print(f"Current steady improvement of {(1-steady_time_clear/steady_time_square)* 100:.2f}% and reset improvement of {(1-reset_time_clear/reset_time_square)* 100:.2f}%")

        num_tries += 1

    print(f"Optimisation completed with steady improvement of {(1-steady_time_clear/steady_time_square)* 100:.2f}% and reset improvement of {(1-reset_time_clear/reset_time_square) * 100:.2f}%")
    print(f"Steady state time CLEAR: {steady_time_clear/1e-9:.2f} ns")
    print(f"Steady state time square: {steady_time_square/1e-9:.2f} ns")
    print(f"Reset time CLEAR: {reset_time_clear/1e-9:.2f} ns")
    print(f"Reset time square: {reset_time_square/1e-9:.2f} ns")

    # ------- UNCOMMENT FOR SEPARATE OPT --------------
    # all_amps = [
    #         steady_params[2],  # ringup1_amp
    #         steady_params[3],  # ringdown1_amp
    #         reset_params[2],   # ringup2_amp
    #         reset_params[3],   # ringdown2_amp
    #     ]

    # ------- UNCOMMENT FOR SIMULTANEOUS OPT --------------
    all_amps = [
            CLEAR_params[2],  # ringup1_amp
            CLEAR_params[3],  # ringdown1_amp
            CLEAR_params[7],   # ringup2_amp
            CLEAR_params[8],   # ringdown2_amp
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

    # ------- UNCOMMENT FOR SEPARATE OPT --------------
    # ringdown1_time = convert_numpy_types(round(steady_params[1]/1e-9))
    # ringup1_time = convert_numpy_types(round(steady_params[0]/1e-9))
    # ringdown2_time = convert_numpy_types(round(reset_params[1]/1e-9))
    # ringup2_time = convert_numpy_types(round(reset_params[0]/1e-9))
    # drive_time = convert_numpy_types(round((pulse_width - steady_params[0] - steady_params[1])/1e-9))

    # ringdown1_amp = convert_numpy_types(steady_params[3] * scale_factor)
    # ringup1_amp = convert_numpy_types(steady_params[2] * scale_factor)
    # ringdown2_amp = convert_numpy_types(reset_params[3] * scale_factor)
    # ringup2_amp = convert_numpy_types(reset_params[2] * scale_factor)
    # drive_amp = convert_numpy_types(steady_params[4] * scale_factor)

    # ------- UNCOMMENT FOR SIMULTANEOUS OPT --------------
    ringdown1_time = convert_numpy_types(round(CLEAR_params[1]/1e-9))
    ringup1_time = convert_numpy_types(round(CLEAR_params[0]/1e-9))
    ringdown2_time = convert_numpy_types(round(CLEAR_params[6]/1e-9))
    ringup2_time = convert_numpy_types(round(CLEAR_params[5]/1e-9))
    drive_time = convert_numpy_types(round((pulse_width - CLEAR_params[0] - CLEAR_params[1])/1e-9))

    ringdown1_amp = convert_numpy_types(CLEAR_params[3] * scale_factor)
    ringup1_amp = convert_numpy_types(CLEAR_params[2] * scale_factor)
    ringdown2_amp = convert_numpy_types(CLEAR_params[8] * scale_factor)
    ringup2_amp = convert_numpy_types(CLEAR_params[7] * scale_factor)
    drive_amp = convert_numpy_types(CLEAR_params[4] * scale_factor)

    length = (ringdown1_time + ringup1_time + ringdown2_time + ringup2_time + drive_time)
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
    print(f"CLEAR pulse params saved to {clear_filepath}")

    plot_optimal_clear(duration, dt, envelope, photon_0, photon_s, env_filepath, photon_filepath)
    print(f"CLEAR pulse figs saved to {env_filepath} and {photon_filepath}")