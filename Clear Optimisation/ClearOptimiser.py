import cma
import random
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def b_in_CLEAR(t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase):
    try:
        if t <= pulse_start:
            return 0.0
        elif t <= pulse_start + ringup1_time:
            return ringup1_amp * np.exp(1j * phase)  # optionally phase
        elif t <= pulse_start + ringup1_time + ringdown1_time:
            return ringdown1_amp * np.exp(1j * phase)
        elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width:
            return drive_amp * np.exp(1j * phase)
        elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width + ringdown2_time:
            return ringdown2_amp * np.exp(1j * phase)
        elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time:
            return ringup2_amp * np.exp(1j * phase)
        else:
            return 0.0
    
    except Exception as e:
        print(f"b_in error at t={t}: {e}")
        return 0.0
    
def b_in_square(t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase):
    try:
        if  t<= pulse_start:
            return 0.0
        elif t <= pulse_start + ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time:
            return drive_amp * np.exp(1j * phase)
        else:   
            return 0.0
    
    except Exception as e:
        print(f"b_in error at t={t}: {e}")
        return 0.0
    

def langevin(t, y,  chi, k, drive_fn, qubit_state):
    if qubit_state == 0: 
        delta = 0
    elif qubit_state == 1:
        delta = +chi
    alpha = y[0] + 1j * y[1]
    d_alpha = -(1j * delta + k/2) * alpha - np.sqrt(k) * drive_fn(t)
    return [d_alpha.real, d_alpha.imag]

def run_langevin(qubit_state, t_span, t_eval, phase, drive, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    
    ringup1_time = 4e-9 * round(ringup1_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    ringdown1_time = 4e-9 * round(ringdown1_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns
    ringup2_time = 4e-9 * round(ringup2_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    ringdown2_time = 4e-9 * round(ringdown2_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns

    b_in = partial(
        drive,
        pulse_start=pulse_start,
        pulse_width=pulse_width,
        ringup1_time=ringup1_time,
        ringdown1_time=ringdown1_time,
        ringup1_amp=ringup1_amp,
        ringdown1_amp=ringdown1_amp,
        drive_amp=drive_amp,
        ringup2_time=ringup2_time,
        ringdown2_time=ringdown2_time,
        ringup2_amp=ringup2_amp,
        ringdown2_amp=ringdown2_amp,
        phase = phase
    )
    
    sol = solve_ivp(langevin, t_span, [0, 0], args=(chi, k, b_in, qubit_state), t_eval=t_eval)
    alpha = sol.y[0] + 1j * sol.y[1]
    return np.array(alpha)


def cost_func(drive, buffer, phase, chi, k, pulse_start, pulse_width, drive_amp, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    
    t_drive = ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time
    t_total = t_drive + buffer
    t_span = (0, t_total)
    t_eval = np.linspace(*t_span, 1000)

    a_g = run_langevin(0, t_span, t_eval, phase, drive, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
    a_e = run_langevin(1, t_span, t_eval, phase, drive, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
    
    b_in_vals = np.array([drive(t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase) for t in t_eval])
    b_out_g = b_in_vals + np.sqrt(k) * a_g
    b_out_e = b_in_vals + np.sqrt(k) * a_e
    diff = b_out_e - b_out_g

    dt = t_eval[1] - t_eval[0]
    integral = np.sum(np.abs(diff)**2) * dt

    cost  = -integral

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval * 1e9, diff, label='Env Diff)', color='blue')
    plt.grid(True)
    plt.title("Envelope difference")
    plt.tight_layout()
    plt.savefig('C:\\Users\\qcrew\\Documents\\jon\\cheddar\\scripts\\CLEAR\\env_diff.png')
    plt.close()

    return cost

class ClearCost:
    def __init__(self, sys_params):
        self.sys_params = sys_params

    def __call__(self, params):
        ringup1_norm = params[2]
        ringdown1_norm = params[3]
        drive_norm = params[4]
        if ringup1_norm <= ringdown1_norm or ringup1_norm <= drive_norm:
            return np.inf
        full_params = np.concatenate((self.sys_params, params))
        return cost_func(*full_params)
    
# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse(buffer, phase, chi, k, pulse_start, pulse_width, drive_amp, best_ringup_params_so_far, best_ringdown_params_so_far, randomise):
    global sys_params_CLEAR
    sys_params_CLEAR = [b_in_CLEAR, buffer, phase, chi, k, pulse_start, pulse_width, drive_amp]
    cost_fn = ClearCost(sys_params_CLEAR)
    best_params = best_ringup_params_so_far + best_ringdown_params_so_far

    N_attempts = 100  # Number of attempts to find a solution
    N_jobs = 10

    params_CLEAR = None

    # Get order of magnitude
    order = int(np.floor(np.log10(abs(drive_amp))))
    base = 10 ** order

    # Define custom ranges
    above_range = (drive_amp * 1.01, base * 20)   # start just above drive_amp
    below_range = (base * 0.01, drive_amp * 0.99)    # start from base, end just below drive_amp

    # Full bounds example
    bounds = [
        (4e-9, 100e-9),           # Ringup1 time
        (4e-9, 100e-9),           # Ringdown1 time
        above_range,              # Ringup1 norm (above drive_amp)
        below_range,              # Ringdown1 norm (below drive_amp)
        (4e-9, 100e-9),           # Ringup2 time
        (4e-9, 100e-9),           # Ringdown2 time
        below_range,              # Ringup2 norm (below drive_amp)
        (-above_range[1], -above_range[0])  # Ringdown2 norm, negative and above -drive_amp
    ]            

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    print("=== Optimising CLEAR ===")
    
    counter = 0
    while params_CLEAR is None:
        if counter > N_attempts:
            print(f"Failed to find a steady_state solution after {N_attempts} attempts. Try adjusting the steady_state threshold.")
            break
        
        # === Initial Guess and Sigma ===
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

        if not best_params:
            x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        elif randomise:
            dev = [(high - low) * 0.4 for (low, high) in bounds]
            ringup = [
                min(max(p + s * random.gauss(0, 1), low), high)
                for (p, s, (low, high)) in zip(best_ringup_params_so_far, dev[0:4], bounds[0:4])
            ]
            ringdown = [
                min(max(p + s * random.gauss(0, 1), low), high)
                for (p, s, (low, high)) in zip(best_ringdown_params_so_far, dev[4:8], bounds[4:8])
            ]
            x0 = ringup + ringdown

        else:
            dev = [(high - low) * 0.08 for (low, high) in bounds]
            x0 = [
                min(max(p + s * random.gauss(0, 1), low), high)
                for (p, s, (low, high)) in zip(best_params, dev, bounds)
            ]

        x0 = [min(max(val, low), high) for val, (low, high) in zip(x0, bounds)]

        # === CMA-ES Optimization ===
        es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=0.3,
            inopts={
                'bounds': [list(lower_bounds), list(upper_bounds)],
                'maxiter': 200,
                'CMA_stds': sigmas,
                'verb_disp': 0,
            }
        )

        # Run the optimization
        res = es.optimize(cost_fn, n_jobs=N_jobs)

        # Best solution and its cost
        # ringup_time, ringdown_time, ringup_norm, ringdown_norm, drive_norm
        if res.result.fbest != np.inf:
            print("Best CLEAR cost:", -res.result.fbest)
        params_CLEAR = res.result.xbest
        counter += 1

    print(f"Stabilisation optimal ringup1_time: {params_CLEAR[0]/1e-9} ns, ringdown1_time {params_CLEAR[1]/1e-9} ns, ringup1_norm {params_CLEAR[2]} V, ringdown1_norm {params_CLEAR[3]}, drive_norm {sys_params_CLEAR[7]}, ringup2_time: {params_CLEAR[4]/1e-9} ns, ringdown2_time {params_CLEAR[5]/1e-9} ns, ringup2_norm {params_CLEAR[6]}, ringdown2_norm {params_CLEAR[7]}")

    return params_CLEAR


def cross_check_with_square(params_CLEAR, buffer, phase, chi, k, pulse_start, pulse_width, drive_amp):
    
    ringup1_amp = params_CLEAR[2]
    ringdown1_amp = params_CLEAR[3]
    ringup2_amp = params_CLEAR[6]
    ringdown2_amp = params_CLEAR[7]

    optimal_ringup1_time = params_CLEAR[0]
    optimal_ringdown1_time = params_CLEAR[1]
    optimal_ringup2_time = params_CLEAR[4]
    optimal_ringdown2_time = params_CLEAR[5]

    optimal_ringup1_time = 4e-9 * round(optimal_ringup1_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    optimal_ringdown1_time = 4e-9 * round(optimal_ringdown1_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns
    optimal_ringup2_time = 4e-9 * round(optimal_ringup2_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    optimal_ringdown2_time = 4e-9 * round(optimal_ringdown2_time / 4e-9)  # Ensure ringdown time is a multiple of 4

    t_drive = optimal_ringup1_time + optimal_ringdown1_time + pulse_width + optimal_ringdown2_time + optimal_ringup2_time
    t_total = t_drive + buffer
    t_span = (0, t_total)
    t_eval = np.linspace(*t_span, 1000)
    dt = t_eval[1] - t_eval[0]

    # Run for both qubit states
    a_s_g = run_langevin(0, t_span, t_eval, phase, b_in_square, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    a_s_e = run_langevin(1, t_span, t_eval, phase, b_in_square, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    b_in_s_vals = np.array([b_in_square(t, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp, phase) for t in t_eval])
    b_out_s_g = b_in_s_vals + np.sqrt(k) * a_s_g
    b_out_s_e = b_in_s_vals + np.sqrt(k) * a_s_e
    diff_s = b_out_s_e - b_out_s_g

    a_c_g = run_langevin(0, t_span, t_eval, phase, b_in_CLEAR, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    a_c_e = run_langevin(1, t_span, t_eval, phase, b_in_CLEAR, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    b_in_c_vals = np.array([b_in_CLEAR(t, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp, phase) for t in t_eval])
    b_out_c_g = b_in_c_vals + np.sqrt(k) * a_c_g
    b_out_c_e = b_in_c_vals + np.sqrt(k) * a_c_e
    diff_c = b_out_c_e - b_out_c_g

    integral_s = np.sum(np.abs(diff_s)**2) * dt
    integral_c = np.sum(np.abs(diff_c)**2) * dt

    return integral_s, integral_c, b_out_s_g, b_out_s_e, b_out_c_g, b_out_c_e, diff_s, diff_c, b_in_c_vals

def plot_optimal_clear(t_eval, envelope, b_out_e, b_out_g, diff_c, diff_s, env_filepath, b_g_filepath, b_e_filepath, diff_filepath):

    # I and Q components
    I_t = np.real(envelope)
    Q_t = np.imag(envelope)

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval * 1e9, I_t, label='I(t)', color='blue')
    plt.plot(t_eval * 1e9, Q_t, label='Q(t)', color='orange')
    plt.title("CLEAR Pulse — I and Q Components")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.savefig(env_filepath)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval * 1e9, np.real(b_out_g), label='Real |g>')
    plt.plot(t_eval * 1e9, np.imag(b_out_g), label='Imag |g>')
    plt.xlabel("Time (ns)")
    plt.ylabel("Amp (arb. units)")
    plt.title("Reflected Field")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(b_g_filepath)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval * 1e9, np.real(b_out_e), label='Real |e>')
    plt.plot(t_eval * 1e9, np.imag(b_out_e), label='Imag |e>')
    plt.xlabel("Time (ns)")
    plt.ylabel("Amp (arb. units)")
    plt.title("Reflected Field")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(b_e_filepath)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval * 1e9, diff_c, label='Diff CLEAR')
    plt.plot(t_eval * 1e9, diff_s, label='Diff Const')
    plt.xlabel("Time (ns)")
    plt.ylabel("Amp (arb. units)")
    plt.title("Field difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(diff_filepath)
    # plt.show()

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj
    
def tune_drive_for_photon(target_n, chi, k):

    delta = +chi

    alpha = np.sqrt(target_n)
    b_in = -alpha * (1j * delta + k / 2) / np.sqrt(k)

    drive_amplitude = np.abs(b_in)

    return drive_amplitude
