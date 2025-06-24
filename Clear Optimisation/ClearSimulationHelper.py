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

def langevin(t, a_vec, chi, k, b_in, qubit_state):
        omega_eff = chi * qubit_state
        return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in(t)

def run_langevin(qubit_state, tlist, dt, chi, k, b_in_func, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    
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
    solver.set_integrator('zvode', method='bdf')  # complex-valued ODE solver
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
def find_steady_state_time(tlist, photon_number, pulse_start, threshold=1e-12, window_size=300):
    start_offset = int(pulse_start* 1e9)
    photon_number = np.array(photon_number[start_offset:])
    tlist = np.array(tlist[start_offset:])

    for i in range(len(photon_number) - window_size):
        window = photon_number[i:i + window_size]
        if np.max(window) - np.min(window) < threshold:
            return tlist[i]
    
    return np.inf

# Ensure that cavity is cleared
def find_cavity_reset_time(tlist, photon_number, pulse_start, pulse_width, threshold=1e-12):
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
def cost_func(mode, duration, dt, chi, k, b_in_func, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp):
    tlist = np.arange(0, duration, dt) 

    a = run_langevin(1, tlist, dt, chi, k, b_in_func, pulse_start, pulse_width, ringup_time, ringdown_time, ringup_amp, ringdown_amp, drive_amp)
    photon = np.abs(a)**2
    if mode == 0:
        steady_state_time = find_cavity_reset_time(tlist, photon, pulse_start, pulse_width)
    elif mode == 1:
        steady_state_time = find_steady_state_time(tlist, photon, pulse_start)
    else:
        raise ValueError("Invalid mode. Use 0 for 'reset' or 1 for 'steady_state'.")

    return steady_state_time/1e-9

# Use evolutionary algorithm to optimize the pulse parameters
def optimise_pulse(duration, dt, chi, k, pulse_start, pulse_width):

    sys_params_rst = [0, duration, dt, chi, k, b_in_ringdown, pulse_start, pulse_width]
    sys_params_steady = [1, duration, dt, chi, k, b_in_ringup, pulse_start, pulse_width]
    optimal_drive = 0

    params_steady = None
    params_reset = None

    def steady_cost(params):
        full_params = np.concatenate((sys_params_steady, params))
        cost = cost_func(*full_params)
        return cost
    
    def reset_cost(params):
        full_params = np.concatenate((sys_params_rst, params, [optimal_drive]))
        cost = cost_func(*full_params)
        return cost

    # === Parameter Bounds ===
    bounds = [(1e-9, 200e-9),       # Ringup time
            (1e-9, 200e-9),         # Ringdown time
            (0.1, 10),              # Ringup norm
            (0.1, 10),              # Ringdown norm
            (0.1, 10)]              # Drive norm

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    while params_steady is None:
        # === Initial Guess and Sigma ===
        x0 = [(low + high) / random.randint(2, 5) for low, high in bounds]  # Midpoint
        # x0 = [10e-9, 50e-9, 10e3, 0.1e3, 2e3]  # Initial guess
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

        # === CMA-ES Optimization ===
        es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=0.3,
            inopts={
                'bounds': [list(lower_bounds), list(upper_bounds)],
                'maxiter': 100,
                'CMA_stds': sigmas,
                'verb_disp': 0,
            }
        )

        # Run the optimization
        res = es.optimize(steady_cost)

        # Best solution and its cost
        # ringup_time, ringdown_time, ringup_norm, ringdown_norm, drive_norm
        print("Best stabilisation time:", res.result.fbest)
        params_steady = res.result.xbest
    print(f"Stabilisation optimal ringup_time: {params_steady[0]/1e-9} ns, ringdown_time {params_steady[1]/1e-9} ns, ringup_norm {params_steady[2]} V, ringdown_norm {params_steady[3]} V, drive_norm {params_steady[4]} V")

    optimal_drive = params_steady[4]

    # === Parameter Bounds ===
    bounds = [(1e-9, 200e-9),       # Ringup time
            (1e-9, 200e-9),         # Ringdown time
            (0.1, 10),              # Ringup norm
            (-10, -0.1),            # Ringdown norm
            ]

    # Extract separate lower and upper bound lists
    lower_bounds, upper_bounds = zip(*bounds)

    while params_reset is None:
        # === Initial Guess and Sigma ===
        x0 = [(low + high) / random.randint(2, 5) for low, high in bounds]  # Midpoint
        # x0 = [50e-9, 50e-9, 1e3, -5e3]  # Initial guess
        sigmas = [(high - low) * 0.3 for (low, high) in bounds]

        # === CMA-ES Optimization ===
        es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=0.3,
            inopts={
                'bounds': [list(lower_bounds), list(upper_bounds)],
                'maxiter': 100,
                'CMA_stds': sigmas,
                'verb_disp': 0,
            }
        )

        # Run the optimization
        res = es.optimize(reset_cost)

        # Best solution and its cost
        # ringup_time, ringdown_time, ringup_norm, ringdown_norm, drive_norm
        print("Best reset time:", res.result.fbest)
        params_reset = res.result.xbest

    params_reset = np.append(params_reset, optimal_drive)

    print(f"Reset optimal ringup_time: {params_reset[0]/1e-9} ns, ringdown_time {params_reset[1]/1e-9} ns, ringup_norm {params_reset[2]} V, ringdown_norm {params_reset[3]} V, drive_norm {params_reset[4]} V")

    return params_steady, params_reset

def plot_optimal_clear(params_steady, params_reset, duration, dt, chi, k, pulse_start, pulse_width):
    ringup1_amp = params_steady[2]
    ringdown1_amp = params_steady[3]
    ringup2_amp = params_reset[2]
    drive_amp = params_steady[4]
    ringdown2_amp = params_reset[3]

    optimal_ringup1_time = params_steady[0]
    optimal_ringdown1_time = params_steady[1]
    optimal_ringup2_time = params_reset[0]
    optimal_ringdown2_time = params_reset[1]

    tlist = np.arange(0, duration, dt) 

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
    def langevin(t, a_vec, qubit_state):
        omega_eff = chi * qubit_state
        return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in(t)

    def langevin_square(t, a_vec, qubit_state):
        omega_eff = chi * qubit_state
        return -1j * omega_eff * a_vec - (k / 2) * a_vec +  1j * np.sqrt(k) * b_in_square(t)

    def run_langevin(qubit_state):
        solver = ode(langevin)
        solver.set_integrator('zvode', method='bdf')  # complex-valued ODE solver
        solver.set_initial_value(0.0+0.0j, tlist[0])
        solver.set_f_params(qubit_state)

        a_vals = []

        for _ in range(len(tlist)):
            if not solver.successful():
                print("Integration failed at t =", solver.t)
                break
            solver.integrate(solver.t + dt)
            a_vals.append(solver.y)

        return np.array(a_vals)

    def run_langevin_square(qubit_state):
        solver = ode(langevin_square)
        solver.set_integrator('zvode', method='bdf')  # complex-valued ODE solver
        solver.set_initial_value(0.0+0.0j, tlist[0])
        solver.set_f_params(qubit_state)

        a_vals = []

        for _ in range(len(tlist)):
            if not solver.successful():
                print("Integration failed at t =", solver.t)
                break
            solver.integrate(solver.t + dt)
            a_vals.append(solver.y)

        return np.array(a_vals)

    # Run for both qubit states
    a_0 = run_langevin(1)
    a_s = run_langevin_square(1)

    photon_0 = np.abs(a_0)**2
    photon_s = np.abs(a_s)**2

    plt.figure(figsize=(10, 5))
    plt.plot(tlist * 1e9, photon_0, label='n [CLEAR]')
    plt.plot(tlist * 1e9, photon_s, label='n [square]')
    plt.xlabel("Time (ns)")
    plt.ylabel("N (arb. units)")
    plt.title("Resonator Photon Number (Complex Langevin Equation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_clear_photon.png")
    plt.show()

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