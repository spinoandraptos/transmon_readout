import re
import numpy as np
from scipy.integrate import solve_ivp

# ---To adhere to OPX clock speed----
def round_to_4(x):
    return 4e-9 * round(x / 4e-9)

# ---Function to evaluate expressions with variables in YAML---
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

class ReadoutSimulator:
    def __init__(
        self,
        ringup1_time: float,
        ringdown1_time: float,
        drive_time: float,
        ringup2_time: float,
        ringdown2_time: float,
        ringup1_amp: float,
        ringdown1_amp: float,
        ringup2_amp: float,
        ringdown2_amp: float,
        attenuation: float,
        kappa: float,
        ramp: float,
        chi: float,
        phase: float,   
        sample_offset_ns: float,
        drive_amp: float,
        offset_r: float = 0.0,
        offset_i: float = 0.0,
        pad: float = 300e-9,
    ):
        # ---------Pulse Parameters----------------------
        self._init_pulse_params(
            ringup1_time, ringdown1_time, drive_time,
            ringup2_time, ringdown2_time,
            ringup1_amp, ringdown1_amp,
            ringup2_amp, ringdown2_amp,
            drive_amp, offset_r, offset_i, pad
        )

        # ---------System Parameters----------------------
        self._init_system_params(attenuation, kappa, chi, phase, ramp)

        # ---------Timing Simulation----------------------
        self._init_timing(sample_offset_ns)

    def _init_pulse_params(
        self, ru1_t, rd1_t, d_t, ru2_t, rd2_t,
        ru1_a, rd1_a, ru2_a, rd2_a,
        drive_amp, offset_r, offset_i, pad
    ):
        self.ringup1_time   = round_to_4(ru1_t)
        self.ringdown1_time = round_to_4(rd1_t)
        self.drive_time     = round_to_4(d_t)
        self.ringup2_time   = round_to_4(ru2_t)
        self.ringdown2_time = round_to_4(rd2_t)

        self.ringup1_amp    = ru1_a
        self.ringdown1_amp  = rd1_a
        self.ringup2_amp    = ru2_a
        self.ringdown2_amp  = rd2_a
        self.drive_amp      = drive_amp

        self.offset_r       = offset_r
        self.offset_i       = offset_i
        self.pad            = pad

    def _init_system_params(self, attenuation, kappa, chi, phase, ramp):
        self.attenuation = attenuation
        self.kappa = kappa * 2 * np.pi 
        self.chi       = chi       * 2 * np.pi 
        self.phase     = phase     * np.pi
        self.ramp      = ramp

    def _init_timing(self, sample_offset_ns):
        self.pulse_start = 0.0
        self.sample_offset_ns = sample_offset_ns

        self.t_drive = (
            self.ringup1_time +
            self.ringdown1_time +
            self.drive_time +
            self.ringdown2_time +
            self.ringup2_time
        )

        self.t_total = self.t_drive + self.sample_offset_ns + self.pad
        self.dt = 1e-9  # 1 ns timestep

        self.sample_interval_ns = 64  # For QCore or similar
        self.t_eval = np.arange(0, self.t_total, self.dt)
        self.t_span = (self.t_eval[0], self.t_eval[-1])

        self.sample_interval_steps = int(self.sample_interval_ns * 1e-9 / self.dt)
        self.sample_offset_steps   = int(self.sample_offset_ns / self.dt)

    # ---------CLEAR Pulse Generation----------------------\
    def smooth_step(self, t, t_start, t_end, rise_time):
        return 0.5 * (np.tanh((t - t_start) / rise_time) - np.tanh((t - t_end) / rise_time))

    def clear_pulse(self, t):

        if self.ramp > 0:
            pulse = 0.0

            # Ringup1
            pulse += self.ringup1_amp * self.smooth_step(t, self.pulse_start, self.pulse_start + self.ringup1_time, self.ramp)

            # Ringdown1
            t1 = self.pulse_start + self.ringup1_time
            t2 = t1 + self.ringdown1_time
            pulse += self.ringdown1_amp * self.smooth_step(t, t1, t2, self.ramp)

            # Drive
            t1 = t2
            t2 = t1 + self.drive_time
            pulse += self.drive_amp * self.smooth_step(t, t1, t2, self.ramp)

            # Ringdown2
            t1 = t2
            t2 = t1 + self.ringdown2_time
            pulse += self.ringdown2_amp * self.smooth_step(t, t1, t2, self.ramp)

            # Ringup2
            t1 = t2
            t2 = t1 + self.ringup2_time
            pulse += self.ringup2_amp * self.smooth_step(t, t1, t2, self.ramp)

            return pulse * np.exp(1j * self.phase)
        
        else:
            if t <= self.pulse_start:
                return 0.0
            elif t <= self.pulse_start + self.ringup1_time:
                return self.ringup1_amp * np.exp(1j * self.phase)
            elif t <= self.pulse_start + self.ringup1_time + self.ringdown1_time:
                return self.ringdown1_amp * np.exp(1j * self.phase)
            elif t <= self.pulse_start + self.ringup1_time + self.ringdown1_time + self.drive_time:
                return self.drive_amp * np.exp(1j * self.phase)
            elif t <= self.pulse_start + self.ringup1_time + self.ringdown1_time + self.drive_time + self.ringdown2_time:
                return self.ringdown2_amp * np.exp(1j * self.phase)
            elif t <= self.pulse_start + self.ringup1_time + self.ringdown1_time + self.drive_time + self.ringdown2_time + self.ringup2_time:
                return self.ringup2_amp * np.exp(1j * self.phase)
            else:
                return 0.0

     # ---------Square Pulse Generation----------------------
    def square_pulse(self, t):
        if self.ramp > 0:
            t0 = self.pulse_start
            t1 = self.pulse_start + self.drive_time
            envelope = self.smooth_step(t, t0, t1, self.ramp)
            return self.drive_amp * envelope * np.exp(1j * self.phase)
        
        else:
            return self.drive_amp * np.exp(1j * self.phase) if self.pulse_start < t < self.drive_time else 0.0
        
    # ---------Semiclassical Langevin Equation----------------------
    def cavity_dynamics(self, t, y, drive_fn, delta):
        alpha = y[0] + 1j * y[1]
        d_alpha = -(1j * delta + self.kappa/2) * alpha - np.sqrt(self.kappa) * drive_fn(t) 
        return [d_alpha.real, d_alpha.imag]

    # ---------Classical Integrator to solve for Alpha----------------------
    def solve_for_state(self, delta):
        sol_clear = solve_ivp(self.cavity_dynamics, self.t_span, [0, 0], args=(self.clear_pulse, delta), t_eval=self.t_eval, method='LSODA')
        alpha_clear = sol_clear.y[0] + 1j * sol_clear.y[1]
        return alpha_clear

    # ---------Used for parameters fitting----------------------
    def get_envelopes(self, factor=0, mode=0):
        
        if mode == 0:
            b_in_vals = np.array([self.clear_pulse(t) for t in self.t_eval])
        else:
            self.t_total = self.drive_time + self.sample_offset_ns + self.pad
            self.t_eval = np.arange(0, self.t_total, self.dt) 
            b_in_vals = np.array([self.square_pulse(t) for t in self.t_eval])

        sol_clear_g = self.solve_for_state(delta=-self.chi * factor)
        sol_clear_e = self.solve_for_state(delta=+self.chi * (1-factor))

        b_out_g = b_in_vals + np.sqrt(self.kappa) * sol_clear_g
        b_out_e = b_in_vals + np.sqrt(self.kappa) * sol_clear_e

        # Sampled time and b_out values
        sample_indices = np.arange(self.sample_offset_steps, len(self.t_eval), self.sample_interval_steps)

        # Sample arrays
        # b_out_g_sampled = b_out_g
        # b_out_e_sampled = b_out_e

        b_out_g_sampled = b_out_g[sample_indices]
        b_out_e_sampled = b_out_e[sample_indices]

        # Offset R and I
        b_out_g_sampled = np.real(b_out_g_sampled)+self.offset_r + 1j*(np.imag(b_out_g_sampled)+self.offset_i)
        b_out_e_sampled = np.real(b_out_e_sampled)+self.offset_r + 1j*(np.imag(b_out_e_sampled)+self.offset_i)

        # sol_g_sampled = sol_clear_g[sample_indices] 
        # sol_e_sampled = sol_clear_e[sample_indices]

        return self.attenuation * b_out_g_sampled, self.attenuation * b_out_e_sampled, sol_clear_g, sol_clear_e
    
    # ---------Used for plotting pulse envelope----------------------
    def get_pulse(self, mode=0):
        if mode == 0:
            return np.array([self.clear_pulse(t) for t in self.t_eval]), self.t_eval
        else:
            self.t_total = self.drive_time + self.sample_offset_ns + self.pad
            self.t_eval = np.arange(0, self.t_total, self.dt)
            return np.array([self.square_pulse(t) for t in self.t_eval]), self.t_eval

    # ---------Cost function used for Optimisation----------------------
    def cost(self, alpha_sep, alpha_clear, alpha_time, alpha_max, factor=0.0, S_min=0.2):

        # --- Solve cavity fields ---
        sol_clear_g = self.solve_for_state(delta=-self.chi * factor)
        sol_clear_e = self.solve_for_state(delta=+self.chi * (1 - factor))

        n_g = np.abs(sol_clear_g)**2
        n_e = np.abs(sol_clear_e)**2

        # --- Output fields for separation ---
        b_in_vals = np.array([self.clear_pulse(t) for t in self.t_eval])
        b_out_g = b_in_vals + np.sqrt(self.kappa) * sol_clear_g
        b_out_e = b_in_vals + np.sqrt(self.kappa) * sol_clear_e

        sample_indices = np.arange(self.sample_offset_steps, len(self.t_eval), self.sample_interval_steps)
        g_s = b_out_g[sample_indices]
        e_s = b_out_e[sample_indices]

        # Normalized separation (0..1-ish)
        num = np.sum(np.abs(e_s - g_s)**2)
        den = np.sum(np.abs(e_s)**2) + np.sum(np.abs(g_s)**2) + 1e-12
        sep_norm = num / den

        # --- Clearing penalty (area under photon number in last tail) ---
        tail = 300
        clear_g = np.trapezoid(n_g[-tail:], dx=self.dt)
        clear_e = np.trapezoid(n_e[-tail:], dx=self.dt)
        clearing_penalty = clear_g + clear_e

        # --- Separation violation (only penalize if too small) ---
        sep_violation = max(0.0, S_min - sep_norm)

        max_ng = np.max(n_g)
        max_ne = np.max(n_e)
        max_n_penalty = max_ne + max_ng

        print(f"Separation: {(sep_violation**2):.2e}, Clearing: {clearing_penalty:.2e}, Max: {max_n_penalty:.2e}, Time: {self.t_drive*1e9:.1f} ns")

        terms = {
            "sep": (sep_violation**2),
            "clear": clearing_penalty,
            "max_n": max_n_penalty

        }

        # --- Weighted cost ---
        cost = (
            + alpha_sep   * terms["sep"]   # maximize separation
            + alpha_clear * terms["clear"] # minimize residual photons
            + alpha_time  * self.t_drive   # minimize total time
            - alpha_max * terms["max_n"]   # minimize maximum photon number
        )

        return cost