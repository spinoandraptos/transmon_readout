import cma
import random
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  
from pathlib import Path

dev_coarse = 0.6  # for exploration
dev_fine = 0.15   # for refinement
sigma = 0.2
SAMPLE_OFS = 0

def scale_clear_params(CLEAR_params, sys_params_CLEAR, MAX_DRIVE):

    highest_drive = max(abs(CLEAR_params[2]), abs(CLEAR_params[3]), abs(CLEAR_params[6]), abs(CLEAR_params[7])) 

    scale_factor = MAX_DRIVE / highest_drive

    CLEAR_params[2] *= scale_factor
    CLEAR_params[3] *= scale_factor
    CLEAR_params[6] *= scale_factor
    CLEAR_params[7] *= scale_factor
    sys_params_CLEAR[6] *= scale_factor


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
            return ringup1_amp * np.exp(1j * phase)
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

def cost_func(drive, buffer, phase, chi, k, pulse_start, drive_amp,
              ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp,
              ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, pulse_width):

    # ---- Time Setup ----
    t_drive = ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time
    t_total = t_drive + buffer
    dt = 1e-9 
    alpha_clear = 1e7
    alpha_time = 1e5

    # Sampling interval
    sample_interval_ns = 64  # in ns
    sample_offset_ns = SAMPLE_OFS       

    t_eval = np.arange(0, t_total, dt)  # ensure inclusive endpoint
    t_span = (t_eval[0], t_eval[-1])        # make sure solve_ivp agrees

    sample_interval_steps = int(sample_interval_ns * 1e-9 / dt)
    sample_offset_steps = int(sample_offset_ns * 1e-9 / dt)
    
    # ---- Langevin Simulation ----
    a_g = run_langevin(0, t_span, t_eval, phase, drive, chi, k,
                       pulse_start, pulse_width,
                       ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp,
                       ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
    
    a_e = run_langevin(1, t_span, t_eval, phase, drive, chi, k,
                       pulse_start, pulse_width,
                       ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp,
                       ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)

    # ---- I/O Fields ----
    b_in_vals = np.array([
        drive(t, pulse_start, pulse_width,
              ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp,
              ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase)
        for t in t_eval
    ])

    b_out_g = b_in_vals + np.sqrt(k) * a_g
    b_out_e = b_in_vals + np.sqrt(k) * a_e
    
    # Sampled time and b_out values
    sample_indices = np.arange(sample_offset_steps, len(t_eval), sample_interval_steps)

    # Sample arrays
    b_out_g_sampled = b_out_g[sample_indices]
    b_out_e_sampled = b_out_e[sample_indices]
    
    separation = np.sum(np.abs(b_out_e_sampled - b_out_g_sampled)**2)

    # --- CLEARING: photon amplitude near end ---
    clear_window_ns = 32  # check last 32 ns
    clear_window_steps = int(clear_window_ns * 1e-9 / dt)
    a_g_clear = np.abs(a_g[-clear_window_steps:])
    a_e_clear = np.abs(a_e[-clear_window_steps:])
    clearing_g = np.mean(a_g_clear**2)
    clearing_e = np.mean(a_e_clear**2)
    clearing_penalty = clearing_g + clearing_e  # penalize residual photons

    # --- PULSE DURATION ---
    duration_penalty = t_drive  # penalize longer drive pulses

    # ---- Weighted Cost ----
    cost = (
        - separation                            # maximize separation
        + alpha_clear * clearing_penalty        # minimize residual photons
        + alpha_time * duration_penalty         # minimize duration
    )

    # print(f"Cost: {cost:.8f}, Separation: {separation:.8f}, Clearing: {alpha_clear * clearing_penalty:.8f}, Duration: {alpha_time * duration_penalty:.5f}")

    return cost

class ClearCost:
    def __init__(self, sys_params, MAX_DRIVE):
        self.sys_params = sys_params
        self.MAX_DRIVE = MAX_DRIVE

    def __call__(self, params):
        
        scale_clear_params(params, self.sys_params, self.MAX_DRIVE)

        full_params = np.concatenate((self.sys_params, params))
        # print(f"Evaluating cost with parameters: {full_params}")
        return cost_func(*full_params)

# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse(buffer, phase, chi, k, pulse_start, drive_amp, best_params,  randomise, MAX_DRIVE, ringup1_range, ringdown1_range, ringup2_range, ringdown2_range, drive_range):
    global sys_params_CLEAR
    sys_params_CLEAR = [b_in_CLEAR, buffer, phase, chi, k, pulse_start, drive_amp]
    cost_fn = ClearCost(sys_params_CLEAR, MAX_DRIVE)

    N_jobs = 10

    params_CLEAR = None

    # Define custom ranges
    above_range = (drive_amp * 1.01, drive_amp * 40,)   # start just above drive_amp
    below_range = (drive_amp  / 40, drive_amp * 0.99)    # start from base, end just below drive_amp

    # Full bounds example
    bounds = [
        ringup1_range,           # Ringup1 time
        ringdown1_range,           # Ringdown1 time
        above_range,              # Ringup1 norm (above drive_amp)
        below_range,              # Ringdown1 norm (below drive_amp)
        ringup2_range,           # Ringup2 time
        ringdown2_range,           # Ringdown2 time
        below_range,              # Ringup2 norm (below drive_amp)
        (-above_range[1], -above_range[0]),  # Ringdown2 norm, negative and above -drive_amp
        drive_range,           # Drive time
    ]            

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    print("=== Optimising CLEAR ===")
    
    def scale_dev(scale):
        return [(high - low) * scale for (low, high) in bounds]
        
    # === Initial Guess and Sigma ===
    sigmas = scale_dev(sigma)

    if len(best_params) == 0:
        # x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        x0 = [
            (low + high)/2 for low, high in bounds
        ]
    elif randomise:
        dev = scale_dev(dev_coarse)
        x0 = [
            min(max(p + s * random.gauss(0, 1), low), high)
            for (p, s, (low, high)) in zip(best_params, dev, bounds)
        ]
    else:
        dev = scale_dev(dev_fine)
        x0 = [
            min(max(p + s * random.gauss(0, 1), low), high)
            for (p, s, (low, high)) in zip(best_params, dev, bounds)
        ]

    # === CMA-ES Optimization ===
    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=0.2,
        inopts={
            'bounds': [list(lower_bounds), list(upper_bounds)],
            'maxiter': 600,
            'CMA_stds': sigmas,
            'verb_disp': 0,
        }
    )

    while not es.stop():
        solutions = es.ask()

        # Evaluate in parallel (adjust n_jobs as needed)
        fitnesses = Parallel(n_jobs=N_jobs)(delayed(cost_fn)(x) for x in solutions)
        es.tell(solutions, fitnesses)
        # es.disp()  # optional: display progress

    print(f"Best fitness: {es.result.fbest:.4f}")
    # print(f"Best parameters: {es.result.xbest}")
    params_CLEAR = es.result.xbest

    best_fitness = es.result.fbest

    # Run the optimization
    # res = es.optimize(cost_fn, n_jobs=N_jobs)

    # print(f"Stabilisation optimal ringup1_time: {params_CLEAR[0]/1e-9} ns, ringdown1_time {params_CLEAR[1]/1e-9} ns, ringup1_norm {params_CLEAR[2]} V, ringdown1_norm {params_CLEAR[3]}, drive_norm {sys_params_CLEAR[6]}, drive_time {params_CLEAR[8]/1e-9},  ringup2_time: {params_CLEAR[4]/1e-9} ns, ringdown2_time {params_CLEAR[5]/1e-9} ns, ringup2_norm {params_CLEAR[6]}, ringdown2_norm {params_CLEAR[7]}")

    return params_CLEAR


def cross_check_with_square(params_CLEAR, buffer, phase, chi, k, pulse_start, drive_amp):
    
    ringup1_amp = params_CLEAR[2]
    ringdown1_amp = params_CLEAR[3]
    ringup2_amp = params_CLEAR[6]
    ringdown2_amp = params_CLEAR[7]

    # print(f"Cross-checking with square pulse: ringup1_amp {ringup1_amp}, ringdown1_amp {ringdown1_amp}, ringup2_amp {ringup2_amp}, ringdown2_amp {ringdown2_amp}")

    optimal_ringup1_time = params_CLEAR[0]
    optimal_ringdown1_time = params_CLEAR[1]
    optimal_ringup2_time = params_CLEAR[4]
    optimal_ringdown2_time = params_CLEAR[5]
    pulse_width = params_CLEAR[8]

    optimal_ringup1_time = 4e-9 * round(optimal_ringup1_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    optimal_ringdown1_time = 4e-9 * round(optimal_ringdown1_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns
    optimal_ringup2_time = 4e-9 * round(optimal_ringup2_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    optimal_ringdown2_time = 4e-9 * round(optimal_ringdown2_time / 4e-9)  # Ensure ringdown time is a multiple of 4
    pulse_width = 4e-9 * round(pulse_width / 4e-9)  # Ensure ringdown time is a multiple of 4

    t_drive = optimal_ringup1_time + optimal_ringdown1_time + pulse_width + optimal_ringdown2_time + optimal_ringup2_time
    t_total = t_drive + buffer
    dt = 1e-9 

    # Sampling interval
    sample_interval_ns = 64  # in ns
    sample_offset_ns = SAMPLE_OFS       

    t_eval = np.arange(0, t_total, dt)  # ensure inclusive endpoint
    t_span = (t_eval[0], t_eval[-1])        # make sure solve_ivp agrees

    sample_interval_steps = int(sample_interval_ns * 1e-9 / dt)
    sample_offset_steps = int(sample_offset_ns * 1e-9 / dt)

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
    
    sample_indices = np.arange(sample_offset_steps, len(t_eval), sample_interval_steps)

    def calc_cost(b_out_e, b_out_g, pulse): 
               
        # Sample arrays
        b_out_g_sampled = b_out_g[sample_indices]
        b_out_e_sampled = b_out_e[sample_indices]

        # # Difference vector
        # diff = b_out_e_sampled - b_out_g_sampled

        # # Matched filter (unit vector in direction of diff)
        # weights = diff / np.linalg.norm(diff)

        # # Hermitian inner product projection
        # iq_e = np.sum(np.conj(weights) * b_out_e_sampled)
        # iq_g = np.sum(np.conj(weights) * b_out_g_sampled)
        
        # # Calculate midpoint
        # midpoint = (iq_e + iq_g) / 2

        # # Shift both points so midpoint is at zero
        # iq_e = iq_e - midpoint
        # iq_g = iq_g - midpoint

        # plt.figure(figsize=(6,6))
        # plt.plot(iq_e.real, iq_e.imag, 'o', label='|e⟩ (Excited)', color='red')
        # plt.plot(iq_g.real, iq_g.imag, 'o', label='|g⟩ (Ground)', color='blue')
        # plt.axhline(0, color='gray', lw=0.5)
        # plt.axvline(0, color='gray', lw=0.5)
        # plt.xlabel('I (In-phase)')
        # plt.ylabel('Q (Quadrature)')
        # plt.title('IQ Separation of Readout Signals')
        # plt.legend()
        # plt.grid(True)
        # plt.axis('equal')  # Equal scaling for x and y axes
        # plt.savefig(str(Path.cwd()) + f"\\{pulse}_IQ.png")
        # plt.close()
        
        # Cost: minimize their overlap
        diff_term = np.sum(np.abs(b_out_e_sampled - b_out_g_sampled)**2)
        cost = diff_term
        return cost
        
    integral_s = calc_cost(b_out_s_e, b_out_s_g, "const")
    integral_c= calc_cost(b_out_c_e, b_out_c_g, "CLEAR")

    return integral_s, integral_c, t_eval[sample_indices], b_out_s_g[sample_indices], b_out_s_e[sample_indices], b_out_c_g[sample_indices], b_out_c_e[sample_indices], diff_s[sample_indices], diff_c[sample_indices], b_in_c_vals[sample_indices], a_c_g[sample_indices], a_c_e[sample_indices], a_s_g[sample_indices], a_s_e[sample_indices]

def plot_optimal_clear(
    t_eval, envelope, 
    b_out_e, b_out_g, 
    diff_c, diff_s,  
    a_c_g, a_c_e, a_s_g, a_s_e, 
    env_filepath, b_g_filepath, b_e_filepath, 
    diff_filepath, a_c_g_filepath, a_c_e_filepath, 
    a_s_g_filepath, a_s_e_filepath
):
    # Convert time to ns
    t_ns = t_eval

    # # Plot I and Q envelope
    # I_t = np.real(envelope)
    # Q_t = np.imag(envelope)
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_ns, I_t, label='I(t)', color='blue')
    # plt.plot(t_ns, Q_t, label='Q(t)', color='orange')
    # plt.title("CLEAR Pulse Envelope — I and Q Components")
    # plt.xlabel("Time (ns)")
    # plt.ylabel("Amplitude (arb. units)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(env_filepath)
    # plt.close()

    # Plot reflected field for |g⟩
    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(b_out_g), label='Real b_out |g⟩', color='blue')
    plt.plot(t_ns, np.imag(b_out_g), label='Imag b_out |g⟩', color='cyan')
    plt.title("Reflected Field (|g⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(b_g_filepath)
    plt.close()

    # Plot reflected field for |e⟩
    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(b_out_e), label='Real b_out |e⟩', color='red')
    plt.plot(t_ns, np.imag(b_out_e), label='Imag b_out |e⟩', color='magenta')
    plt.title("Reflected Field (|e⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(b_e_filepath)
    plt.close()

    # Plot field difference
    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.abs(diff_c), label='|Diff CLEAR|', color='green')
    plt.plot(t_ns, np.abs(diff_s), label='|Diff Square|', color='black', linestyle='--')
    plt.title("Difference in Reflected Fields (|e⟩ - |g⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(env_filepath)
    plt.close()

    # Plot cavity field for CLEAR pulse (|g⟩ and |e⟩)
    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(a_c_g), label='Real α_c |g⟩', color='blue')
    plt.plot(t_ns, np.imag(a_c_g), label='Imag α_c |g⟩', color='cyan', linestyle='--')
    plt.title("Cavity Field (CLEAR |g⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(a_c_g_filepath)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(a_c_e), label='Real α_c |e⟩', color='red')
    plt.plot(t_ns, np.imag(a_c_e), label='Imag α_c |e⟩', color='magenta', linestyle='--')
    plt.title("Cavity Field (CLEAR |e⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(a_c_e_filepath)
    plt.close()

    # Plot cavity field for square pulse (|g⟩ and |e⟩)
    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(a_s_g), label='Real α_s |g⟩', color='blue')
    plt.plot(t_ns, np.imag(a_s_g), label='Imag α_s |g⟩', color='cyan', linestyle='--')
    plt.title("Cavity Field (Square |g⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(a_s_g_filepath)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, np.real(a_s_e), label='Real α_s |e⟩', color='red')
    plt.plot(t_ns, np.imag(a_s_e), label='Imag α_s |e⟩', color='magenta', linestyle='--')
    plt.title("Cavity Field (Square |e⟩)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (arb. units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(a_s_e_filepath)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(t_ns, np.real(b_out_g), label='CLEAR R (|g⟩)', color='blue')
    ax1.plot(t_ns, np.imag(b_out_g), label='CLEAR I (|g⟩)', color='orange')
    ax1.plot(t_ns, np.real(b_out_e), label='CLEAR R (|e⟩)', color='blue', linestyle='--')
    ax1.plot(t_ns, np.imag(b_out_e), label='CLEAR I (|e⟩)', color='orange', linestyle='--')
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Field amplitude")
    plt.suptitle("Return Signal Dynamics (Sampled Every 64 ns)")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(diff_filepath)
    plt.close()
    
    # Difference vector
    diff = b_out_e - b_out_g

    # Matched filter (unit vector in direction of diff)
    weights = diff / np.linalg.norm(diff)

    # Hermitian inner product projection
    iq_e = np.sum(np.conj(weights) * b_out_e)
    iq_g = np.sum(np.conj(weights) * b_out_g)
    
    # Calculate midpoint
    midpoint = (iq_e + iq_g) / 2

    # Shift both points so midpoint is at zero
    iq_e = iq_e - midpoint
    iq_g = iq_g - midpoint

    plt.figure(figsize=(6,6))
    plt.plot(iq_e.real, iq_e.imag, 'o', label='|e⟩ (Excited)', color='red')
    plt.plot(iq_g.real, iq_g.imag, 'o', label='|g⟩ (Ground)', color='blue')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel('I (In-phase)')
    plt.ylabel('Q (Quadrature)')
    plt.title('IQ Separation of Readout Signals')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.savefig(str(Path.cwd()) + f"\\CLEAR_IQ.png")
    plt.close()

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



# # Difference vector
# diff = b_out_e_sampled - b_out_g_sampled

# # Matched filter (unit vector in direction of diff)
# weights = diff / np.linalg.norm(diff)

# # Hermitian inner product projection
# iq_e = np.sum(np.conj(weights) * b_out_e_sampled)
# iq_g = np.sum(np.conj(weights) * b_out_g_sampled)

# # Calculate midpoint
# midpoint = (iq_e + iq_g) / 2

# # Shift both points so midpoint is at zero
# iq_e = iq_e - midpoint
# iq_g = iq_g - midpoint

# # Cost: minimize their overlap
# cost = np.real(iq_e) - np.real(iq_g)