import cma
import random
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from qutip import *
from pathlib import Path

# Hilbert space dimensions
N = 25                        # max photon number in cavity

# Operators
# Define quantum operators
a = tensor(destroy(N), qeye(2))    # Resonator lowering operator
q = tensor(qeye(N), destroy(2))    # Qubit lowering operator
# Define qubit Pauli operators in composite space
sx = tensor(qeye(N), sigmax())
sy = tensor(qeye(N), sigmay())
sz = tensor(qeye(N), sigmaz())

# Initial state: cavity vacuum
psi0 = tensor(basis(N, 0), basis(2,0))  # Vacuum state for resonator, ground state for qubit
psi1 = tensor(basis(N, 0), basis(2,1))  # Vacuum state for resonator, excited state for qubit

def get_photon_trace(mode, qubit_state, tlist, chi, k, T1, T2, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):    
    
    def b_in(t):
        try:
            if  t<= pulse_start:
                return 0.0
            elif t<= pulse_start + ringup1_time:
                return ringup1_amp 
            elif t <= pulse_start + ringup1_time + ringdown1_time:
                return ringdown1_amp
            elif t <= pulse_start + pulse_width:
                return drive_amp
            elif t <= pulse_start + pulse_width + ringdown2_time:
                return ringdown2_amp 
            elif t <= pulse_start + pulse_width + ringdown2_time + ringup2_time:
                return ringup2_amp
            else:   
                return 0.0
        
        except Exception as e:
            print(f"b_in error at t={t}: {e}")
            return 0.0
        
    def b_in_square(t, args):
        state = args['state']  # ±1
        if  t<= pulse_start:
            return 0.0
        elif t <= pulse_start + pulse_width:
            return drive_amp* np.exp(-1j * chi * state * t)
        else:   
            return 0.0

    # Drive pulse envelope
    def b_in_CLEAR(t, args):
        state = args['state']  # ±1
        if  t<= pulse_start:
            return 0.0
        elif t<= pulse_start + ringup1_time:
            return ringup1_amp * np.exp(-1j * chi * state * t)
        elif t <= pulse_start + ringup1_time + ringdown1_time:
            return ringdown1_amp * np.exp(-1j * chi * state * t)
        elif t <= pulse_start + pulse_width:
            return drive_amp * np.exp(-1j * chi * state * t)
        elif t <= pulse_start + pulse_width + ringdown2_time:
            return ringdown2_amp  * np.exp(-1j * chi * state * t)
        elif t <= pulse_start + pulse_width + ringdown2_time + ringup2_time:
            return ringup2_amp * np.exp(-1j * chi * state * t)
        else:   
            return 0.0
        
    if mode == 0:
        epsilon = b_in_square
    elif mode == 1:
        epsilon = b_in_CLEAR

    args_g = {'state': -1}
    args_e = {'state': 1}

    gamma1 = 1.0 / T1
    gamma2 = 1.0 / T2
    gamma_phi = gamma2 - gamma1/2
    gamma_phi = max(gamma_phi, 0)  # avoid negative due to finite T1, T2

    H0 = chi * a.dag() * a * q.dag() * q
    H_drive = [a.dag(), epsilon]  # a^\dagger * epsilon(t)
    H_drive_conj = [a, lambda t, args: np.conj(epsilon(t, args))]  # a * epsilon*(t)
    H = [H0, H_drive, H_drive_conj]

    # Collapse operator for cavity decay
    c_ops = [np.sqrt(k) * a]

    if gamma1 > 0:
        # print("T1 in simulation")
        c_ops.append(np.sqrt(gamma1) * q)

    if gamma_phi > 0:
        # print("T2 in simulation")
        c_ops.append(np.sqrt(gamma_phi) * sz)

    e_ops = [a.dag() * a] 
    epsilon_ss = drive_amp  # constant amplitude in rad/s

    if qubit_state == 1:
        psi = psi1
        args = args_g
        # For |e⟩ state: q†q = 1 ⇒ H0 = chi * a†a
        H_ss = chi * a.dag() * a + epsilon_ss * a.dag() + np.conj(epsilon_ss) * a
    elif qubit_state == 0:
        psi = psi0
        args = args_e
        # For |g⟩ state: q†q = 0 ⇒ H0 = 0
        H_ss = epsilon_ss * a.dag() + np.conj(epsilon_ss) * a

    # Solve master equation
    result = mesolve(H, psi, tlist, c_ops, e_ops, args=args)
    n_t = result.expect[0]
    # Compute steady state
    rho_ss = steadystate(H_ss, c_ops)

    # Compute steady-state photon number
    n_ss = expect(a.dag() * a, rho_ss)

    # Pulse envelope shape
    pulse_shape = np.array([b_in(t) for t in tlist])
    pulse_scale = n_ss / np.max(pulse_shape)
    pulse_shape_scaled = pulse_shape * pulse_scale

    # Plot photon number
    plt.plot(tlist * 1e9, n_t, label="|g>")
    plt.axhline(n_ss, color='orange', linestyle='--', alpha=0.6, label='n_ss |g⟩')
    plt.plot(tlist * 1e9, pulse_shape_scaled, 'k--', alpha=0.6, label='Drive envelope (scaled)')
    plt.xlabel("Time (ns)")
    plt.ylabel("Photon number")
    plt.title("Photon number vs time (QuTiP)")
    plt.grid(True)
    plt.legend() 

    plt.savefig(str(Path.cwd()) + f"\\{qubit_state}_photon.png")
    plt.close()

    return n_t

# Ensure that steady state is maintained for at least 300ns
def find_steady_state_time(tlist, photon_number, pulse_start, pulse_width=None, threshold=1e-9, window_size=300):
    start_offset = int(pulse_start* 1e9)
    photon_number = np.array(photon_number[start_offset:])
    tlist = np.array(tlist[start_offset:])
    
    if pulse_width is None:
        for i in range(len(photon_number) - window_size):
            window = photon_number[i:i + window_size]
            if np.max(window) - np.min(window) < threshold:
                return tlist[i]
    else:
        end_offset = int((pulse_start + pulse_width) * 1e9)
        photon_number = photon_number[:end_offset - start_offset]
        tlist = tlist[:end_offset - start_offset]

        for i in range(len(photon_number)):
            window = photon_number[i:]
            if np.max(window) - np.min(window) < threshold:
                return tlist[i]
    
    return np.inf

# Ensure that cavity is cleared
def find_cavity_reset_time(tlist, photon_number, pulse_start, pulse_width, threshold=1e-9):
    offset = int((pulse_start + pulse_width) * 1e9)
    photon_number = np.array(photon_number[offset:])
    tlist = np.array(tlist[offset:])

    below = photon_number <= threshold

    for i in range(len(below)):
        if np.all(below[i:]):
            # Reset time is relative to when the pulse ends
            return tlist[i] - (pulse_start + pulse_width)

    return np.inf  # No such reset point found

def cost_func_simul(verbose, qubit_state, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady, threshold_reset, drive_amp, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    tlist = np.arange(0, duration, dt) 

    photon = get_photon_trace(1, qubit_state, tlist, chi, k, T1, T2, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
    
    reset_state_time = find_cavity_reset_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, pulse_width=pulse_width, threshold=threshold_reset)
    steady_state_time = find_steady_state_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, pulse_width=pulse_width, threshold=threshold_steady)
    cost  = 0.7 * steady_state_time + 0.3 * reset_state_time
    if verbose:
        print(f"Steady time: {steady_state_time/1e-9}")
        print(f"Reset time: {reset_state_time/1e-9}")

        plt.figure(figsize=(10, 5))
        plt.plot(tlist/1e-9, photon, label='n [CLEAR]')
        plt.xlabel("Time (ns)")
        plt.ylabel("N (arb. units)")
        plt.title("Resonator Photon Number (Complex Langevin Equation)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return cost/1e-9

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
        return cost_func_simul(*full_params)

# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse_simul(verbose, qubit_state, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady, threshold_reset, drive_amp, best_ringup_params_so_far, best_ringdown_params_so_far, randomise_steady, randomise_reset):
    global sys_params_CLEAR
    sys_params_CLEAR = [verbose, qubit_state, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady, threshold_reset, drive_amp]
    cost_fn = ClearCost(sys_params_CLEAR)
    best_params = best_ringup_params_so_far + best_ringdown_params_so_far

    N_attempts = 1000  # Number of attempts to find a solution
    N_explore = 20  # Number of exploration attempts before increasing the threshold
    N_jobs = 10

    params_CLEAR = None
    # Get order of magnitude
    order = int(np.floor(np.log10(abs(drive_amp))))
    base = 10 ** order

    # Define custom ranges
    above_range = (drive_amp * 1.01, base * 20)   # start just above drive_amp, up to 1.5e8
    below_range = (base * 0.1, drive_amp * 0.99)    # start from base, end just below drive_amp

    # Full bounds example
    bounds = [
        (4e-9, 240e-9),           # Ringup1 time
        (4e-9, 240e-9),           # Ringdown1 time
        above_range,              # Ringup1 norm (above drive_amp)
        below_range,              # Ringdown1 norm (below drive_amp)
        (4e-9, 240e-9),           # Ringup2 time
        (4e-9, 240e-9),           # Ringdown2 time
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
        elif counter % N_explore == 0 and counter > 0:
            sys_params_CLEAR[10] = sys_params_CLEAR[10] * 10
            sys_params_CLEAR[11] = sys_params_CLEAR[11] * 10
            print(f"Steady_state threshold increased to {sys_params_CLEAR[10]}")
            print(f"Reset_state threshold increased to {sys_params_CLEAR[11]}")

        
        # === Initial Guess and Sigma ===
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

        if not best_params:
            x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        elif randomise_steady or randomise_reset:
            dev = [(high - low) * 0.4 for (low, high) in bounds]
            ringup = [
                min(max(p + s * random.gauss(0, 1), low), high)
                for (p, s, (low, high)) in zip(best_ringup_params_so_far, dev[0:4], bounds[0:4])
            ]
            ringdown = [
                min(max(p + s * random.gauss(0, 1), low), high)
                for (p, s, (low, high)) in zip(best_ringdown_params_so_far, dev[4:8], bounds[4:8])
            ]
            if randomise_steady and randomise_reset:
                x0 = ringup + ringdown
            elif randomise_steady:
                # ringup = [random.uniform(low, high) for low, high in bounds[0:4]]
                x0 = ringup + best_ringdown_params_so_far
            elif randomise_reset:
                # ringdown = [random.uniform(low, high) for low, high in bounds[4:8]]
                x0 = best_ringup_params_so_far + ringdown
        # elif counter == 0:
        #     x0 = best_params
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

        print("New opt")
        # Run the optimization
        res = es.optimize(cost_fn, n_jobs=N_jobs)

        # Best solution and its cost
        # ringup_time, ringdown_time, ringup_norm, ringdown_norm, drive_norm
        if res.result.fbest != np.inf:
            print("Best CLEAR cost:", res.result.fbest)
        params_CLEAR = res.result.xbest
        counter += 1

    print(f"Stabilisation optimal ringup1_time: {params_CLEAR[0]/1e-9} ns, ringdown1_time {params_CLEAR[1]/1e-9} ns, ringup1_norm {params_CLEAR[2]} V, ringdown1_norm {params_CLEAR[3]} V, drive_norm {sys_params_CLEAR[12]} V, ringup2_time: {params_CLEAR[4]/1e-9} ns, ringdown2_time {params_CLEAR[5]/1e-9} ns, ringup2_norm {params_CLEAR[6]} V, ringdown2_norm {params_CLEAR[7]} V, steady_state threshold {sys_params_CLEAR[10]}, reset threshold {sys_params_CLEAR[11]}")

    return params_CLEAR, sys_params_CLEAR[10], sys_params_CLEAR[11]

def cross_check_with_square_simul(qubit_state, params_CLEAR, duration, dt, chi, k, T1, T2, pulse_start, pulse_width, threshold_steady, threshold_reset, drive_amp):
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

    tlist = np.arange(0, duration, dt)
    thresh_steady = threshold_steady
    thresh_reset = threshold_reset

    # Define drive
    def b_in(t):
        try:
            if  t<= pulse_start:
                return 0.0
            elif t<= pulse_start + optimal_ringup1_time:
                return ringup1_amp 
            elif t <= pulse_start + optimal_ringup1_time + optimal_ringdown1_time:
                return ringdown1_amp
            elif t <= pulse_start + pulse_width:
                return drive_amp
            elif t <= pulse_start + pulse_width + optimal_ringdown2_time:
                return ringdown2_amp 
            elif t <= pulse_start + pulse_width + optimal_ringdown2_time + optimal_ringup2_time:
                return ringup2_amp
            else:   
                return 0.0
        
        except Exception as e:
            print(f"b_in error at t={t}: {e}")
            return 0.0
        
    photon_0 = get_photon_trace(1, qubit_state, tlist, chi, k, T1, T2, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    photon_s = get_photon_trace(0,qubit_state, tlist, chi, k, T1, T2, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
    envelope = np.array([b_in(t) for t in tlist.flatten()])
    envelope = np.array([b_in(t) for t in tlist.flatten()])

    steady_time_clear = find_steady_state_time(tlist=tlist, photon_number=photon_0, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_steady)
    steady_time_square = find_steady_state_time(tlist=tlist, photon_number=photon_s, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_steady)
    while steady_time_square == np.inf:
        thresh_steady *= 1.5
        steady_time_square = find_steady_state_time(tlist=tlist, photon_number=photon_s, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_steady)
        # print(f"LATEST STEADY SQUARE: {steady_time_square} at threshold {thresh_steady}")
    print(f"Final Steady Threshold: {thresh_steady}")
    reset_time_clear = find_cavity_reset_time(tlist=tlist, photon_number=photon_0, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_reset)
    reset_time_square = find_cavity_reset_time(tlist=tlist, photon_number=photon_s, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_reset)
    while reset_time_square == np.inf:
        thresh_reset *= 1.1
        reset_time_square = find_cavity_reset_time(tlist=tlist, photon_number=photon_s, pulse_start=pulse_start, pulse_width=pulse_width, threshold=thresh_reset)
        # print(f"LATEST RESET SQUARE: {reset_time_square} at threshold {thresh_reset}")
    print(f"Final Reset Threshold: {thresh_reset}")
    return steady_time_clear, steady_time_square, reset_time_clear, reset_time_square, envelope, photon_0, photon_s

def plot_optimal_clear(duration, dt, envelope, photon_0, photon_s, env_filepath, photon_filepath):
    tlist = np.arange(0, duration, dt)
    # I and Q components
    I_t = np.real(envelope)
    Q_t = np.imag(envelope)

    plt.figure(figsize=(10, 5))
    plt.plot(tlist/1e-9, I_t, label='I(t)', color='blue')
    plt.plot(tlist/1e-9, Q_t, label='Q(t)', color='orange')
    plt.title("CLEAR Pulse — I and Q Components")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.savefig(env_filepath)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(tlist/1e-9, photon_0, label='n [CLEAR]')
    plt.plot(tlist/1e-9, photon_s, label='n [square]')
    plt.xlabel("Time (ns)")
    plt.ylabel("N (arb. units)")
    plt.title("Resonator Photon Number (Complex Langevin Equation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(photon_filepath)
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
    
def get_photon(drive_amp, qubit_state, chi, k, T1, T2):

    epsilon_ss = drive_amp  # constant amplitude in rad/s

    gamma1 = 1.0 / T1
    gamma2 = 1.0 / T2
    gamma_phi = gamma2 - gamma1/2
    gamma_phi = max(gamma_phi, 0)  # avoid negative due to finite T1, T2

    if qubit_state == 1:
        # For |e⟩ state: q†q = 1 ⇒ H0 = chi * a†a
        H_ss = chi * a.dag() * a + epsilon_ss * a.dag() + np.conj(epsilon_ss) * a
    elif qubit_state == 0:
        # For |g⟩ state: q†q = 0 ⇒ H0 = 0
        H_ss = epsilon_ss * a.dag() + np.conj(epsilon_ss) * a

    # Collapse operator for cavity decay
    c_ops = [np.sqrt(k) * a]

    if gamma1 > 0:
        c_ops.append(np.sqrt(gamma1) * q)

    if gamma_phi > 0:
        c_ops.append(np.sqrt(gamma_phi) * sz)

    # Compute steady state
    rho_ss = steadystate(H_ss, c_ops)

    # Compute steady-state photon number
    n_ss = expect(a.dag() * a, rho_ss)

    return n_ss

# Binary search for drive amplitude that yields photon number target
def tune_drive_for_photon(verbose, qubit_state, chi, k, T1, T2, target_photon=3.6, lower=0.0,  upper=1e4, tol=1e-3, max_iter=50):
    for _ in range(max_iter):
        mid = (lower + upper) / 2
        photon_val = get_photon(mid, qubit_state, chi, k, T1, T2)
        if verbose:
            print(f"Drive = {mid:.4f}, Photon = {photon_val:.4f}")
        if abs(photon_val - target_photon) < tol:
            return mid
        elif photon_val < target_photon:
            lower = mid
        else:
            upper = mid
    raise RuntimeError("Did not converge within max iterations")






