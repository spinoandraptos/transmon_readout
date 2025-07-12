""" """
import numpy as np
from config.experiment_config import FOLDER, N, I, Q, SINGLE_SHOT
from qcore import Experiment, qua, Sweep
from qm import qua as qm_qua
from qcore.libs.qua_macros import QuaVariable

qubit = "B"

class ResidualPhoton(Experiment):
    """Residual Photon Measurement post Pulse"""

    primary_datasets = ["I", "Q", "single_shot"]
    primary_sweeps = ["time_delay"]

    def sequence(self):

        # Virtual detuning factor
        factor = qm_qua.declare(qm_qua.fixed)
        qm_qua.assign(factor, self.detuning * 1e-9)

        # Play CLEAR Pulse
        # self.resonator.play(self.clear_readout_pulse)
        self.resonator.play(self.clear_readout_pulse)
        qua.align(self.qubit, self.resonator)

        # Sweep the time after CLEAR pulse for extracting residual photon number after pulse
        qua.wait(self.relax_time, self.qubit, self.resonator)

        qua.reset_frame(self.qubit)
        self.qubit.play(self.qubit_drive) # pi/2

        qua.wait(self.time_delay / 2, self.qubit)  # Half wait
        self.qubit.play(self.echo_pulse)
        qua.wait(self.time_delay / 2, self.qubit)  # Half wait

        qm_qua.assign(self.phase, qm_qua.Cast.mul_fixed_by_int(factor, self.time_delay))
        self.qubit.play(self.qubit_drive, phase=self.phase) # pi/2
        qua.align()

        self.resonator.measure(
            self.readout_pulse, (self.I, self.Q), ampx=self.ro_ampx, demod_type="dual"
        )
        qua.wait(self.wait_time, self.resonator)
        qm_qua.assign(
            self.single_shot,
            qm_qua.Cast.to_fixed(self.I > self.readout_pulse.threshold),
        )

if __name__ == "__main__":
    """ """

    modes = {
        "qubit": f"q{qubit}",
        "resonator": f"rr{qubit}",
    }

    pulses = {
        "qubit_drive": f"q{qubit}_gaussian_pi2_pulse",
        "echo_pulse": f"q{qubit}_gaussian_pi_pulse",
        "readout_pulse": f"rr{qubit}_readout_pulse",
        "clear_readout_pulse": f"rr{qubit}_CLEAR_readout_pulse",
    }

    N.num = 500

    # Delay 16 to 30000 ns for free evolution
    DEL = Sweep(name="time_delay", start=16, stop=30_000, step=160, dtype=int)

    sweeps = [N, DEL]

    I.fitfn = "exp_decay_sine"
    I.plot = False
    Q.fitfn = "exp_decay_sine"
    Q.plot = False
    SINGLE_SHOT.fitfn = "exp_decay_sine"
    SINGLE_SHOT.plot = False
    datasets = [I, Q,SINGLE_SHOT]

    # Delay 4 to 1000 ns after pulse for residual photon decay
    points = np.linspace(4, 1000, 30)
    times = np.round(points).astype(int) * 4

    for relax_time in times:
        print(f"Experiment: Residual photons after {relax_time/4} ns") 
        parameters = {
            "relax_time": relax_time,
            "wait_time": 500_000,
            "ro_ampx": 1,
            "detuning": 1e6,
            "phase": QuaVariable(
                value=0.0,
                dtype=qm_qua.fixed,
                tag="phase",
                buffer=True,
                stream=True,
            ),
            "number_of_echo": 3,
        }

        expt = ResidualPhoton(FOLDER, modes, pulses, sweeps, datasets, **parameters)
        expt.run()
