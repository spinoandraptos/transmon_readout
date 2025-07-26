import numpy as np
from scipy.integrate import solve_ivp

def round_to_4(x):
    return 4e-9 * round(x / 4e-9)

class ReadoutSimulator():
    def __init__(
        self,
        kappa,   
        chi,
        phase,
        drive_amp,
        ringup1_time,
        ringdown1_time,
        drive_time,
        ringup2_time,
        ringdown2_time,
        ringup1_amp,
        ringdown1_amp,
        ringup2_amp,
        ringdown2_amp,
    ):
        self.kappa = kappa
        self.chi = chi
        self.ringup1_time = round_to_4(ringup1_time)
        self.ringdown1_time = round_to_4(ringdown1_time)
        self.drive_time = round_to_4(drive_time)
        self.ringup2_time = round_to_4(ringup2_time)
        self.ringdown2_time = round_to_4(ringdown2_time)
        self.ringup1_amp = ringup1_amp
        self.ringdown1_amp = ringdown1_amp
        self.drive_amp = drive_amp
        self.ringup2_amp = ringup2_amp
        self.ringdown2_amp = ringdown2_amp
        self.phase = phase
        self.buffer = 0.0
        self.pulse_start = 0.0
        self.t_drive = ringup1_time + ringdown1_time + drive_time + ringdown2_time + ringup2_time
        self.t_total = self.t_drive + self.buffer
        self.dt = 1e-9  
        self.sample_interval_ns = 64  # in ns
        self.sample_offset_ns = 0          
        self.t_eval = np.arange(0, self.t_total, self.dt) 
        self.t_span = (self.t_eval[0], self.t_eval[-1])      
        self.sample_interval_steps = int(self.sample_interval_ns * 1e-9 / self.dt)
        self.sample_offset_steps = int(self.sample_offset_ns * 1e-9 / self.dt)

    def clear_pulse(self, t):
        if t <= self.pulse_start:
            return 0.0
        elif t <= self.pulse_start + self.ringup1_time:
            return self.ringup1_amp * np.exp(1j * self.phase)  # optionally phase
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

    def square_pulse(self, t):
        return self.drive_amp * np.exp(1j * self.phase) if 0 < t < self.t_drive else 0.0

    def cavity_dynamics(self, t, y, drive_fn, delta):
        alpha = y[0] + 1j * y[1]
        d_alpha = -(1j * delta + self.kappa/2) * alpha - np.sqrt(self.kappa) * drive_fn(t)
        return [d_alpha.real, d_alpha.imag]

    def solve_for_state(self, delta):
        sol_clear = solve_ivp(self.cavity_dynamics, self.t_span, [0, 0], args=(self.clear_pulse, delta), t_eval=self.t_eval)
        alpha_clear = sol_clear.y[0] + 1j * sol_clear.y[1]
        return alpha_clear

    def cost(self, alpha_clear, alpha_time):
        sol_clear_g = self.solve_for_state(delta=0)
        sol_clear_e = self.solve_for_state(delta=+self.chi)

        b_in_vals = np.array([self.clear_pulse(t) for t in self.t_eval])
        b_out_g = b_in_vals + np.sqrt(self.kappa) * sol_clear_g
        b_out_e = b_in_vals + np.sqrt(self.kappa) * sol_clear_e

        # Sampled time and b_out values
        sample_indices = np.arange(self.sample_offset_steps, len(self.t_eval), self.sample_interval_steps)

        # Sample arrays
        b_out_g_sampled = b_out_g[sample_indices] 
        b_out_e_sampled = b_out_e[sample_indices]

        separation = np.sum(np.abs(b_out_e_sampled - b_out_g_sampled)**2)

        # --- CLEARING: photon amplitude near end ---
        clearing_g = np.real(sol_clear_g[-1])**2 + np.imag(sol_clear_g[-1])**2
        clearing_e = np.real(sol_clear_e[-1])**2 + np.imag(sol_clear_e[-1])**2
        clearing_penalty = clearing_g + clearing_e  # penalize residual photons

        # --- PULSE DURATION ---
        duration_penalty = self.t_drive  # penalize longer drive pulses

        # ---- Weighted Cost ----
        cost = (
            - separation                            # maximize separation
            + alpha_clear * clearing_penalty        # minimize residual photons
            + alpha_time * duration_penalty         # minimize duration
        )

        return cost