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
        sample_offset_ns,
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
        pad=30e-9,
        offset_r = 0.0,
        offset_i = 0.0
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
        self.pulse_start = 0.0
        self.phase = phase * np.pi
        self.pad = pad
        self.t_drive = ringup1_time + ringdown1_time + drive_time + ringdown2_time + ringup2_time
        self.sample_offset_ns = sample_offset_ns      
        self.t_total = self.t_drive + self.sample_offset_ns + self.pad
        self.dt = 1e-9  
        self.sample_interval_ns = 64  # in ns
        self.t_eval = np.arange(0, self.t_total, self.dt) 
        self.t_span = (self.t_eval[0], self.t_eval[-1])      
        self.sample_interval_steps = int(self.sample_interval_ns * 1e-9 / self.dt)
        self.sample_offset_steps = int(self.sample_offset_ns / self.dt)
        self.offset_r = offset_r
        self.offset_i = offset_i

    def clear_pulse(self, t):
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

    # def smooth_transition(self, t, t_start, duration, amp_start, amp_end, phase):
    #     """
    #     Smoothly interpolates between amp_start and amp_end over [t_start, t_start+duration]
    #     without dipping below min(amp_start, amp_end).
    #     """
    #     if t < t_start:
    #         return amp_start * np.exp(1j * phase)
    #     elif t > t_start + duration:
    #         return amp_end * np.exp(1j * phase)

    #     tau = (t - t_start) / duration  # normalized time 0 to 1
    #     s = 0.5 * (1 - np.cos(np.pi * tau))  # smooth monotonic in [0,1]

    #     amp = (1 - s) * amp_start + s * amp_end
    #     return amp * np.exp(1j * phase)


    # def clear_pulse(self, t):
    #     t0 = self.pulse_start
    #     t1 = t0 + self.ringup1_time
    #     t2 = t1 + self.ringdown1_time
    #     t3 = t2 + self.drive_time
    #     t4 = t3 + self.ringdown2_time
    #     t5 = t4 + self.ringup2_time

    #     if t < t0 or t > t5:
    #         return 0.0
    #     elif t <= t1:
    #         return self.smooth_transition(t, t0, self.ringup1_time, 0.0, self.ringup1_amp, self.phase)
    #     elif t <= t2:
    #         return self.smooth_transition(t, t1, self.ringdown1_time, self.ringup1_amp, self.ringdown1_amp, self.phase)
    #     elif t <= t3:
    #         return self.smooth_transition(t, t2, self.drive_time, self.ringdown1_amp, self.drive_amp, self.phase)
    #     elif t <= t4:
    #         return self.smooth_transition(t, t3, self.ringdown2_time, self.drive_amp, self.ringdown2_amp, self.phase)
    #     elif t <= t5:
    #         return self.smooth_transition(t, t4, self.ringup2_time, self.ringdown2_amp, self.ringup2_amp, self.phase)


    def square_pulse(self, t):
        return self.drive_amp * np.exp(1j * self.phase) if 0 < t < self.t_drive else 0.0

    def cavity_dynamics(self, t, y, drive_fn, delta):
        alpha = y[0] + 1j * y[1]
        d_alpha = -(1j * delta + self.kappa/2) * alpha - np.sqrt(self.kappa) * drive_fn(t)
        # print(f"t: {t}, alpha: {alpha}, d_alpha: {d_alpha}")  # Debugging line
        return [d_alpha.real, d_alpha.imag]

    def solve_for_state(self, delta):
        sol_clear = solve_ivp(self.cavity_dynamics, self.t_span, [self.offset_r, self.offset_i], args=(self.clear_pulse, delta), t_eval=self.t_eval)
        alpha_clear = sol_clear.y[0] + 1j * sol_clear.y[1]
        return alpha_clear

    def cost(self, alpha_clear, alpha_time, factor=0.0):

        sol_clear_g = self.solve_for_state(delta=-self.chi * factor)
        sol_clear_e = self.solve_for_state(delta=+self.chi * (1-factor))

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
            - separation                  # maximize separation
            + alpha_clear * clearing_penalty        # minimize residual photons
            + alpha_time * duration_penalty         # minimize duration
        )

        return cost
    
    def get_envelopes(self, factor=0):
        sol_clear_g = self.solve_for_state(delta=-self.chi * factor)
        sol_clear_e = self.solve_for_state(delta=+self.chi * (1-factor))

        b_in_vals = np.array([self.clear_pulse(t) for t in self.t_eval])
        b_out_g = b_in_vals + np.sqrt(self.kappa) * sol_clear_g
        b_out_e = b_in_vals + np.sqrt(self.kappa) * sol_clear_e

        # Sampled time and b_out values
        sample_indices = np.arange(self.sample_offset_steps, len(self.t_eval), self.sample_interval_steps)

        # Sample arrays
        b_out_g_sampled = b_out_g[sample_indices] 
        b_out_e_sampled = b_out_e[sample_indices]

        return b_out_g_sampled, b_out_e_sampled
    
    def get_pulse(self):
        return np.array([self.clear_pulse(t) for t in self.t_eval]), self.t_eval