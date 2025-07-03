import cma
import random
import numpy as np
from functools import partial
from scipy.integrate import ode
import matplotlib.pyplot as plt

def b_in_ringup(t, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    try:
        if  t<= pulse_start:
            return 0.0
        elif t<= pulse_start + ringup_time:
            return ringup_amp 
        elif t <= pulse_start + ringup_time + ringdown_time:
            return ringdown_amp
        else:
            return drive_amp

    except Exception as e:
        print(f"b_in error at t={t}: {e}")
        return 0.0
    
def b_in_ringdown(t, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    try:
        if t <= pulse_start + pulse_width:
            return drive_amp
        elif t <= pulse_start + pulse_width + ringdown_time:
            return ringdown_amp 
        elif t <= pulse_start + pulse_width + ringdown_time + ringup_time:
            return ringup_amp
        else:   
            return 0.0
    
    except Exception as e:
        print(f"b_in error at t={t}: {e}")
        return 0.0
    
def b_in_CLEAR(t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
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
    
def b_in_square(t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    try:
        if  t<= pulse_start:
            return 0.0
        elif t <= pulse_start + pulse_width:
            return drive_amp
        else:   
            return 0.0
    
    except Exception as e:
        print(f"b_in error at t={t}: {e}")
        return 0.0
    
def langevin(t, a_vec, chi, k, b_in, qubit_state):
        omega_eff = chi * qubit_state
        return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in(t)

def run_langevin(qubit_state, tlist, dt, chi, k, b_in_func, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    
    ringup_time = 4e-9 * round(ringup_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    ringdown_time = 4e-9 * round(ringdown_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns
    
    b_in = partial(
        b_in_func,
        pulse_start=pulse_start,
        pulse_width=pulse_width,
        ringup_time=ringup_time,
        ringdown_time=ringdown_time,
        ringup_amp=ringup_amp,
        ringdown_amp=ringdown_amp,
        drive_amp=drive_amp
    )
    
    solver = ode(langevin)
    solver.set_integrator('zvode', method='bdf', max_step=dt)  # complex-valued ODE solver
    solver.set_initial_value(0.0+0.0j, tlist[0])
    solver.set_f_params(chi, k, b_in, qubit_state)
    a_vals = []

    for _ in range(len(tlist)):
        if not solver.successful():
            print("Integration failed at t =", solver.t)
            break
        solver.integrate(solver.t + dt)
        a_vals.append(solver.y)

    return np.array(a_vals)

def run_langevin_simul(qubit_state, tlist, dt, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    
    ringup1_time = 4e-9 * round(ringup1_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    ringdown1_time = 4e-9 * round(ringdown1_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns
    ringup2_time = 4e-9 * round(ringup2_time / 4e-9)  # Ensure ringup time is a multiple of 4ns
    ringdown2_time = 4e-9 * round(ringdown2_time / 4e-9)  # Ensure ringdown time is a multiple of 4ns

    b_in = partial(
        b_in_CLEAR,
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
        ringdown2_amp=ringdown2_amp
    )
    
    solver = ode(langevin)
    solver.set_integrator('zvode', method='bdf', max_step=dt)  # complex-valued ODE solver
    solver.set_initial_value(0.0+0.0j, tlist[0])
    solver.set_f_params(chi, k, b_in, qubit_state)
    a_vals = []

    for _ in range(len(tlist)):
        if not solver.successful():
            print("Integration failed at t =", solver.t)
            break
        solver.integrate(solver.t + dt)
        a_vals.append(solver.y)

    return np.array(a_vals)

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

# Cost functuion for optimization
def cost_func(mode, duration, dt, chi, k, b_in_func, pulse_start, pulse_width, threshold, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    tlist = np.arange(0, duration, dt) 

    a = run_langevin(1, tlist, dt, chi, k, b_in_func, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp)
    photon = np.abs(a)**2
    if mode == 0:
        steady_state_time = find_cavity_reset_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, pulse_width=pulse_width, threshold=threshold)
    elif mode == 1:
        steady_state_time = find_steady_state_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, threshold=threshold)
    else:
        raise ValueError("Invalid mode. Use 0 for 'reset' or 1 for 'steady_state'.")

    return steady_state_time/1e-9

def cost_func_simul(duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
    tlist = np.arange(0, duration, dt) 

    a = run_langevin_simul(1, tlist, dt, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
    photon = np.abs(a)**2
    reset_state_time = find_cavity_reset_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, pulse_width=pulse_width, threshold=threshold_reset)
    steady_state_time = find_steady_state_time(tlist=tlist, photon_number=photon, pulse_start=pulse_start, pulse_width=pulse_width, threshold=threshold_steady)
    cost  = 0.6 * steady_state_time + 0.4 * reset_state_time
    # print(f"Steady time: {steady_state_time/1e-9}")
    # print(f"Reset time: {reset_state_time/1e-9}")

    return cost/1e-9

def steady_cost(params):
    global sys_params_steady
    ringup_norm = params[2]
    ringdown_norm = params[3]
    drive_norm = params[4]
    if ringup_norm <= ringdown_norm or ringup_norm <= drive_norm or ringdown_norm >= drive_norm:
        return np.inf
    full_params = np.concatenate((sys_params_steady, params))
    cost = cost_func(*full_params)
    return cost

def reset_cost(params):
    global sys_params_rst
    ringup_norm = params[2]
    if ringup_norm >= optimal_drive:
        return np.inf
    full_params = np.concatenate((sys_params_rst, params, [optimal_drive]))
    cost = cost_func(*full_params)
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
        return cost_func_simul(*full_params)

# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse(duration, dt, chi, k, pulse_start, pulse_width, threshold_steady=1e-9, threshold_reset=1e-9):
    global sys_params_rst
    global sys_params_steady
    global optimal_drive

    N_attempts = 1000  # Number of attempts to find a solution
    N_explore = 200  # Number of exploration attempts before increasing the threshold
    N_jobs = 6

    sys_params_rst = [0, duration, dt, chi, k, b_in_ringdown, pulse_start, pulse_width, threshold_reset]
    sys_params_steady = [1, duration, dt, chi, k, b_in_ringup, pulse_start, pulse_width, threshold_steady]
    optimal_drive = 0

    params_steady = None
    params_reset = None


    # === Parameter Bounds ===
    bounds = [(4e-9, 24e-9),       # Ringup time (Minimum QUA resolution is 4ns, 1 cycle)
            (4e-9, 24e-9),         # Ringdown time (Minimum QUA resolution is 4ns, 1 cycle)
            (1e3, 15e3),              # Ringup norm
            (1e3, 5e3),              # Ringdown norm
            (1e3, 10e3),
            ]              # Drive norm

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    print("=== Optimising Steady State ===")
    
    counter = 0
    while params_steady is None:
        if counter > N_attempts:
            print(f"Failed to find a steady_state solution after {N_attempts} attempts. Try adjusting the steady_state threshold.")
            break
        elif counter % N_explore == 0 and counter > 0:
            sys_params_steady[8] = sys_params_steady[8] * 10
            print(f"Steady_state threshold increased to {sys_params_steady[8]}")

        
        # === Initial Guess and Sigma ===
        x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

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
        res = es.optimize(steady_cost, n_jobs=N_jobs)

        # Best solution and its cost
        # ringup_time, ringdown_time, ringup_norm, ringdown_norm, drive_norm
        if res.result.fbest != np.inf:
            print("Best stabilisation time:", res.result.fbest)
        params_steady = res.result.xbest
        counter += 1

    print(f"Stabilisation optimal ringup_time: {params_steady[0]/1e-9} ns, ringdown_time {params_steady[1]/1e-9} ns, ringup_norm {params_steady[2]} V, ringdown_norm {params_steady[3]} V, drive_norm {params_steady[4]} V, steady_state threshold {sys_params_steady[8]}")

    optimal_drive = params_steady[4]

    # === Parameter Bounds ===
    bounds = [(4e-9, 24e-9),       # Ringup time (Minimum QUA resolution is 4ns, 1 cycle)
            (4e-9, 24e-9),         # Ringdown time (Minimum QUA resolution is 4ns, 1 cycle)
            (1e3, 5e3),              # Ringup norm
            (-15e3, -1e3),            # Ringdown norm
            ]

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    print("=== Optimising Reset ===")

    counter = 0
    while params_reset is None:
        if counter > N_attempts:
            print(f"Failed to find a reset solution after {N_attempts} attempts. Try adjusting the reset threshold.")
            break
        elif counter % N_explore == 0 and counter > 0:
            sys_params_rst[8] = sys_params_rst[8] * 10
            print(f"Reset threshold increased to {sys_params_rst[8]}")

        # === Initial Guess and Sigma ===
        x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

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
        res = es.optimize(reset_cost, n_jobs=N_jobs)

        # Best solution and its cost
        if res.result.fbest != np.inf:
            print("Best reset time:", res.result.fbest)
        params_reset = res.result.xbest
        counter += 1

    params_reset = np.append(params_reset, optimal_drive)

    print(f"Reset optimal ringup_time: {params_reset[0]/1e-9} ns, ringdown_time {params_reset[1]/1e-9} ns, ringup_norm {params_reset[2]} V, ringdown_norm {params_reset[3]} V, drive_norm {params_reset[4]} V, reset threshold {sys_params_rst[8]}")

    return params_steady, params_reset, sys_params_steady[8], sys_params_rst[8]

# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse_simul(duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset):
    global sys_params_CLEAR
    sys_params_CLEAR = [duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset]
    cost_fn = ClearCost(sys_params_CLEAR)

    N_attempts = 1000  # Number of attempts to find a solution
    N_explore = 100  # Number of exploration attempts before increasing the threshold
    N_jobs = 6

    params_CLEAR = None

    # === Parameter Bounds ===
    bounds = [(100e-9, 400e-9),        # Ringup1 time (Minimum QUA resolution is 4ns, 1 cycle)
            (100e-9, 400e-9),          # Ringdown1 time (Minimum QUA resolution is 4ns, 1 cycle)
            (1e3, 15e3),            # Ringup1 norm
            (1e3, 5e3),             # Ringdown1 norm
            (1e3, 10e3),            # Drive norm
            (100e-9, 400e-9),          # Ringup2 time (Minimum QUA resolution is 4ns, 1 cycle)
            (100e-9, 400e-9),          # Ringdown2 time (Minimum QUA resolution is 4ns, 1 cycle)
            (1e3, 5e3),             # Ringup2 norm
            (-15e3, -1e3),          # Ringdown2 norm
        ]                           

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    print("=== Optimising Steady State ===")
    
    counter = 0
    while params_CLEAR is None:
        if counter > N_attempts:
            print(f"Failed to find a steady_state solution after {N_attempts} attempts. Try adjusting the steady_state threshold.")
            break
        elif counter % N_explore == 0 and counter > 0:
            sys_params_CLEAR[6] = sys_params_CLEAR[6] * 10
            sys_params_CLEAR[7] = sys_params_CLEAR[7] * 10
            print(f"Steady_state threshold increased to {sys_params_CLEAR[6]}")
            print(f"Reset_state threshold increased to {sys_params_CLEAR[7]}")

        
        # === Initial Guess and Sigma ===
        x0 = [random.uniform(low, high) for low, high in bounds]  # Midpoint
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

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
            print("Best CLEAR cost:", res.result.fbest)
        params_CLEAR = res.result.xbest
        counter += 1

    print(f"Stabilisation optimal ringup1_time: {params_CLEAR[0]/1e-9} ns, ringdown1_time {params_CLEAR[1]/1e-9} ns, ringup1_norm {params_CLEAR[2]} V, ringdown1_norm {params_CLEAR[3]} V, drive_norm {params_CLEAR[4]} V, ringup2_time: {params_CLEAR[5]/1e-9} ns, ringdown2_time {params_CLEAR[6]/1e-9} ns, ringup2_norm {params_CLEAR[7]} V, ringdown2_norm {params_CLEAR[8]} V, steady_state threshold {sys_params_CLEAR[6]}, reset threshold {sys_params_CLEAR[7]}")

    return params_CLEAR, sys_params_CLEAR[6], sys_params_CLEAR[7]


def cross_check_with_square(params_steady, params_reset, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset):
    ringup1_amp = params_steady[2]
    ringdown1_amp = params_steady[3]
    ringup2_amp = params_reset[2]
    drive_amp = params_steady[4]
    ringdown2_amp = params_reset[3]

    optimal_ringup1_time = params_steady[0]
    optimal_ringdown1_time = params_steady[1]
    optimal_ringup2_time = params_reset[0]
    optimal_ringdown2_time = params_reset[1]

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
        
    def b_in_square(t):
        try:
            if  t<= pulse_start:
                return 0.0
            elif t <= pulse_start + pulse_width:
                return drive_amp
            else:   
                return 0.0
        
        except Exception as e:
            print(f"b_in error at t={t}: {e}")
            return 0.0

    # Define Heisenburg picture 
    def langevin(t, a_vec, qubit_state, mode):
        omega_eff = chi * qubit_state
        if mode == 0:
            return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in(t)
        elif mode == 1:
            return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in_square(t)

    def run_langevin(qubit_state, mode):
        solver = ode(langevin)
        solver.set_integrator('zvode', method='bdf', max_step=dt)  # complex-valued ODE solver
        solver.set_initial_value(0.0+0.0j, tlist[0])
        solver.set_f_params(qubit_state, mode)

        a_vals = []

        for _ in range(len(tlist)):
            if not solver.successful():
                print("Integration failed at t =", solver.t)
                break
            solver.integrate(solver.t + dt)
            a_vals.append(solver.y)

        return np.array(a_vals)

    # Run for both qubit states
    a_0 = run_langevin(1, 0)
    a_s = run_langevin(1, 1)

    photon_0 = np.abs(a_0)**2
    photon_s = np.abs(a_s)**2
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


def cross_check_with_square_simul(params_CLEAR, duration, dt, chi, k, pulse_start, pulse_width, threshold_steady, threshold_reset):
    ringup1_amp = params_CLEAR[2]
    ringdown1_amp = params_CLEAR[3]
    ringup2_amp = params_CLEAR[7]
    drive_amp = params_CLEAR[4]
    ringdown2_amp = params_CLEAR[8]

    optimal_ringup1_time = params_CLEAR[0]
    optimal_ringdown1_time = params_CLEAR[1]
    optimal_ringup2_time = params_CLEAR[5]
    optimal_ringdown2_time = params_CLEAR[6]

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
        
    def b_in_square(t):
        try:
            if  t<= pulse_start:
                return 0.0
            elif t <= pulse_start + pulse_width:
                return drive_amp
            else:   
                return 0.0
        
        except Exception as e:
            print(f"b_in error at t={t}: {e}")
            return 0.0

    # Define Heisenburg picture 
    def langevin(t, a_vec, qubit_state, mode):
        omega_eff = chi * qubit_state
        if mode == 0:
            return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in(t)
        elif mode == 1:
            return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in_square(t)

    def run_langevin(qubit_state, mode):
        solver = ode(langevin)
        solver.set_integrator('zvode', method='bdf', max_step=dt)  # complex-valued ODE solver
        solver.set_initial_value(0.0+0.0j, tlist[0])
        solver.set_f_params(qubit_state, mode)

        a_vals = []

        for _ in range(len(tlist)):
            if not solver.successful():
                print("Integration failed at t =", solver.t)
                break
            solver.integrate(solver.t + dt)
            a_vals.append(solver.y)

        return np.array(a_vals)

    # Run for both qubit states
    a_0 = run_langevin(1, 0)
    a_s = run_langevin(1, 1)

    photon_0 = np.abs(a_0)**2
    photon_s = np.abs(a_s)**2
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