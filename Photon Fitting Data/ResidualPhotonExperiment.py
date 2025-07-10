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
        self.resonator.play(self.clear_readout_pulse)

        # Sweep the time after CLEAR pulse for extracting residual photon number after pulse
        qua.wait(self.relax_time, self.resonator)

        # Reset qubit frame for virtual detuning
        qua.reset_frame(self.qubit)

        # Bring qubit to superposition with pi/2
        self.qubit.play(self.qubit_drive) 

        # Half wait free evolution virtual phase accumulation
        qua.wait(self.ramsey_time / 2, self.qubit)
        
        # State inversion with pi to negate effect of slow noise 
        self.qubit.play(self.echo_pulse)

        # Half wait free evolution virtual phase accumulation
        qua.wait(self.ramsey_time / 2, self.qubit)

        # Project back to measurement basis with accumulated virtual phase
        qm_qua.assign(self.phase, qm_qua.Cast.mul_fixed_by_int(factor, self.time_delay))
        self.qubit.play(self.qubit_drive, phase=self.phase) # pi/2
        qua.align()

        # Determine excited state population
        self.resonator.measure(
            self.readout_pulse, (self.I, self.Q), ampx=self.ro_ampx, demod_type="dual"
        )
        qua.wait(self.wait_time, self.resonator)
        if self.plot_single_shot:  # assign state to G or E
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
    RAMSEY_SWEEP = Sweep(name="ramsey_time", start=16, stop=30_000, step=160, dtype=int)

    sweeps = [N, RAMSEY_SWEEP]

    I.fitfn = "exp_decay_sine"
    I.plot = False
    Q.fitfn = "exp_decay_sine"
    Q.plot = False
    SINGLE_SHOT.fitfn = "exp_decay_sine"
    SINGLE_SHOT.plot = False
    datasets = [I, Q,SINGLE_SHOT]

    # Delay 4 to 1000 ns after pulse for residual photon decay
    points = np.linspace(4, 1000, 30)
    int_points = np.round(points).astype(int)

    for val in int_points:
        print(f"Experiment: Residual photons after {val} ns")
        parameters = {
            "relax_time": val,
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
            "plot_single_shot": False,
            "number_of_echo": 3,
        }

        expt = ResidualPhoton(FOLDER, modes, pulses, sweeps, datasets, **parameters)
        expt.run()
