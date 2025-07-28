import cma
import random
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  

class ClearOptimiser():

    def __init__(
        self,
        dev_refine,
        sigma_cma,
        sampling_offset,
        weight_penalise_photon,
        weight_penalise_length,
    ) -> None:
        self.dev_fine = dev_refine
        self.sigma = sigma_cma
        self.SAMPLE_OFS = sampling_offset
        self.alpha_clear = weight_penalise_photon
        self.alpha_time = weight_penalise_length

    def scale_clear_params(self, CLEAR_params, sys_params_CLEAR, MAX_DRIVE):

        highest_drive = max(abs(CLEAR_params[2]), abs(CLEAR_params[3]), abs(CLEAR_params[6]), abs(CLEAR_params[7])) 

        scale_factor = MAX_DRIVE / highest_drive

        CLEAR_params[2] *= scale_factor
        CLEAR_params[3] *= scale_factor
        CLEAR_params[6] *= scale_factor
        CLEAR_params[7] *= scale_factor
        sys_params_CLEAR[6] *= scale_factor

    def smooth_cosine_edge(self, t, t0, duration, amp, phase):
        """ Smooth cosine-squared ramp between 0 and amp over `duration` starting at `t0`. """
        if t < t0 or t > t0 + duration:
            return 0.0
        tau = (t - t0) / duration
        window = np.sin(np.pi * tau / 2)**2  # cosine-squared ramp
        return amp * window * np.exp(1j * phase)

    def b_in_CLEAR(self, t, pulse_start, pulse_width,
                ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp,
                drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase):
        try:
            t0 = pulse_start
            t1 = t0 + ringup1_time
            t2 = t1 + ringdown1_time
            t3 = t2 + pulse_width
            t4 = t3 + ringdown2_time
            t5 = t4 + ringup2_time

            if t < t0:
                return 0.0
            elif t <= t1:
                return self.smooth_cosine_edge(t, t0, ringup1_time, ringup1_amp, phase)
            elif t <= t2:
                return self.smooth_cosine_edge(t, t1, ringdown1_time, ringdown1_amp, phase)
            elif t <= t3:
                return drive_amp * np.exp(1j * phase)
            elif t <= t4:
                return self.smooth_cosine_edge(t, t3, ringdown2_time, ringdown2_amp, phase)
            elif t <= t5:
                return self.smooth_cosine_edge(t, t4, ringup2_time, ringup2_amp, phase)
            else:
                return 0.0

        except Exception as e:
            print(f"b_in error at t={t}: {e}")
            return 0.0
        
    def b_in_square(self, t, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, phase):
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
        

    def langevin(self, t, y,  chi, k, drive_fn, qubit_state):
        if qubit_state == 0: 
            delta = 0
        elif qubit_state == 1:
            delta = +chi
        alpha = y[0] + 1j * y[1]
        d_alpha = -(1j * delta + k/2) * alpha - np.sqrt(k) * drive_fn(t)
        return [d_alpha.real, d_alpha.imag]

    def run_langevin(self, qubit_state, t_span, t_eval, phase, drive, chi, k, pulse_start, pulse_width, ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp):
        
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

        sol = solve_ivp(self.langevin, t_span, [0, 0], args=(chi, k, b_in, qubit_state), t_eval=t_eval)
        alpha = sol.y[0] + 1j * sol.y[1]
        
        return np.array(alpha)


    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
        
    def tune_drive_for_photon(self, target_n, chi, k):

        delta = +chi

        alpha = np.sqrt(target_n)
        b_in = -alpha * (1j * delta + k / 2) / np.sqrt(k)

        drive_amplitude = np.abs(b_in)

        return drive_amplitude


    def cost_func(self, drive, buffer, phase, chi, k, pulse_start, drive_amp,
                ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp,
                ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp, pulse_width):

        # ---- Time Setup ----
        t_drive = ringup1_time + ringdown1_time + pulse_width + ringdown2_time + ringup2_time
        t_total = t_drive + buffer
        dt = 1e-9 

        # Sampling interval
        sample_interval_ns = 64  # in ns
        sample_offset_ns = self.SAMPLE_OFS       

        t_eval = np.arange(0, t_total, dt)  # ensure inclusive endpoint
        t_span = (t_eval[0], t_eval[-1])        # make sure solve_ivp agrees

        sample_interval_steps = int(sample_interval_ns * 1e-9 / dt)
        sample_offset_steps = int(sample_offset_ns * 1e-9 / dt)
        
        # ---- Langevin Simulation ----
        a_g = self.run_langevin(0, t_span, t_eval, phase, drive, chi, k,
                        pulse_start, pulse_width,
                        ringup1_time, ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp,
                        ringup2_time, ringdown2_time, ringup2_amp, ringdown2_amp)
        
        a_e = self.run_langevin(1, t_span, t_eval, phase, drive, chi, k,
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

        weights = np.linspace(10, 0.1, len(b_out_e_sampled))  # Heavier at start
        separation = np.sum(weights * np.abs(b_out_e_sampled - b_out_g_sampled)**2)

        # --- CLEARING: photon amplitude near end ---
        clear_window_ns = 32  # check last 32 ns
        clear_window_steps = int(clear_window_ns * 1e-9 / dt)
        clearing_g = np.mean(np.real(a_g[-clear_window_steps:])**2) + np.mean(np.imag(a_g[-clear_window_steps:])**2)
        clearing_e = np.mean(np.real(a_e[-clear_window_steps:])**2) + np.mean(np.imag(a_e[-clear_window_steps:])**2)
        clearing_penalty = clearing_g + clearing_e  # penalize residual photons

        # --- PULSE DURATION ---
        duration_penalty = t_drive  # penalize longer drive pulses

        # ---- Weighted Cost ----
        cost = (
            - separation                            # maximize separation
            + self.alpha_clear * clearing_penalty        # minimize residual photons
            + self.alpha_time * duration_penalty         # minimize duration
        )

        return cost
    
    class ClearCost:
        def __init__(self, sys_params, MAX_DRIVE, optimiser):
            self.sys_params = sys_params
            self.MAX_DRIVE = MAX_DRIVE
            self.optimiser = optimiser

        def __call__(self, params):
            
            self.optimiser.scale_clear_params(params, self.sys_params, self.MAX_DRIVE)

            full_params = np.concatenate((self.sys_params, params))
            return self.optimiser.cost_func(*full_params)

    # Use evolutionary algorithm to optimize the pulse parameters
    def optimise_pulse(self, buffer, phase, chi, k, pulse_start, drive_amp, best_params,  randomise, MAX_DRIVE, ringup1_range, ringdown1_length_range, ringup2_length_range, ringdown2_length_range, drive_length_range, ringup1_amp_range, ringdown1_amp_range, ringup2_amp_range, ringdown2_amp_range):
        global sys_params_CLEAR
        sys_params_CLEAR = [self.b_in_CLEAR, buffer, phase, chi, k, pulse_start, drive_amp]
        cost_fn = self.ClearCost(sys_params_CLEAR, MAX_DRIVE, self)

        N_jobs = 10

        params_CLEAR = None

        # Define custom ranges
        ringup1_amp_range = (drive_amp * ringup1_amp_range[0], drive_amp * ringup1_amp_range[1])          # start just above drive_amp
        ringdown1_amp_range = (drive_amp * ringdown1_amp_range[0], drive_amp * ringdown1_amp_range[1])       # end just below drive_amp
        ringup2_amp_range = (drive_amp * ringup2_amp_range[0], drive_amp * ringup2_amp_range[1])
        ringdown2_amp_range = (drive_amp * ringdown2_amp_range[0], drive_amp * ringdown2_amp_range[1])

        # Full bounds example
        bounds = [
            ringup1_range,                                              # Ringup1 time
            ringdown1_length_range,                                     # Ringdown1 time
            ringup1_amp_range,                                          # Ringup1 norm 
            ringdown1_amp_range,                                        # Ringdown1 norm 
            ringup2_length_range,                                       # Ringup2 time
            ringdown2_length_range,                                     # Ringdown2 time
            ringup2_amp_range,                                          # Ringup2 norm
            ringdown2_amp_range,                                        # Ringdown2 norm, negative 
            drive_length_range,                                         # Drive time
        ]            

        # Extract separate lower and upper bound lists
        lower_bounds, upper_bounds = zip(*bounds)

        print("=== Optimising CLEAR ===")
        
        def scale_dev(scale):
            return [(high - low) * scale for (low, high) in bounds]
            
        # === Initial Guess and Sigma ===
        sigmas = scale_dev(self.sigma)

        if len(best_params) == 0:
            # x0 = [random.uniform(low, high) for low, high in bounds]  
            x0 = [
                (low + high)/2 for low, high in bounds # Midpoint
            ]
        elif randomise:
            # dev = scale_dev(dev_coarse)
            # x0 = [
            #     min(max(p + s * random.gauss(0, 1), low), high)
            #     for (p, s, (low, high)) in zip(best_params, dev, bounds)
            # ]
            x0 = [random.uniform(low, high) for low, high in bounds]
        else:
            dev = scale_dev(self.dev_fine)
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
            fitnesses = Parallel(n_jobs=N_jobs)(delayed(cost_fn)(x) for x in solutions)
            es.tell(solutions, fitnesses)

        params_CLEAR = es.result.xbest
        best_fitness = es.result.fbest
        print(f"Best fitness: {best_fitness:.4f}")

        return params_CLEAR, -best_fitness


    def generate_vis(self, params_CLEAR, buffer, phase, chi, k, pulse_start, drive_amp, MAX_DRIVE):
        
        params_CLEAR_copy = params_CLEAR.copy()
        sys_params_CLEAR = [0,0,0,0,0,0, drive_amp] 
        self.scale_clear_params(params_CLEAR_copy, sys_params_CLEAR, MAX_DRIVE)
        drive_amp = sys_params_CLEAR[6]

        ringup1_amp = params_CLEAR_copy[2]
        ringdown1_amp = params_CLEAR_copy[3]
        ringup2_amp = params_CLEAR_copy[6]
        ringdown2_amp = params_CLEAR_copy[7]

        optimal_ringup1_time = params_CLEAR_copy[0]
        optimal_ringdown1_time = params_CLEAR_copy[1]
        optimal_ringup2_time = params_CLEAR_copy[4]
        optimal_ringdown2_time = params_CLEAR_copy[5]
        pulse_width = params_CLEAR_copy[8]

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
        sample_offset_ns = self.SAMPLE_OFS       

        t_eval = np.arange(0, t_total, dt)  # ensure inclusive endpoint
        t_span = (t_eval[0], t_eval[-1])        # make sure solve_ivp agrees

        sample_interval_steps = int(sample_interval_ns * 1e-9 / dt)
        sample_offset_steps = int(sample_offset_ns * 1e-9 / dt)

        # Run for both qubit states
        a_c_g = self.run_langevin(0, t_span, t_eval, phase, self.b_in_CLEAR, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
        a_c_e = self.run_langevin(1, t_span, t_eval, phase, self.b_in_CLEAR, chi, k, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp)
        b_in_c_vals = np.array([self.b_in_CLEAR(t, pulse_start, pulse_width, optimal_ringup1_time, optimal_ringdown1_time, ringup1_amp, ringdown1_amp, drive_amp, optimal_ringup2_time, optimal_ringdown2_time, ringup2_amp, ringdown2_amp, phase) for t in t_eval])
        b_out_c_g = b_in_c_vals + np.sqrt(k) * a_c_g
        b_out_c_e = b_in_c_vals + np.sqrt(k) * a_c_e 
        
        sample_indices = np.arange(sample_offset_steps, len(t_eval), sample_interval_steps)

        return t_eval[sample_indices], b_out_c_g[sample_indices], b_out_c_e[sample_indices], b_in_c_vals[sample_indices], a_c_g[sample_indices], a_c_e[sample_indices]

    def plot_optimal_clear(
        self, t_eval, envelope, 
        b_out_e, b_out_g,  
        a_c_g, a_c_e,
        env_filepath, 
        diff_filepath, a_c_g_filepath, a_c_e_filepath, 
    ):
        # Convert time to ns
        t_ns = t_eval

        # Plot I and Q envelope
        I_t = np.real(envelope)
        Q_t = np.imag(envelope)
        plt.figure(figsize=(10, 4))
        plt.plot(t_ns, I_t, label='Envelope', color='blue')
        plt.plot(t_ns, Q_t, label='Q(t)', color='orange')
        plt.title("CLEAR Pulse Envelope")
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

        _, ax1 = plt.subplots(figsize=(7, 5))
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

