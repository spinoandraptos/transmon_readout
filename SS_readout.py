""" """
import sys
from config.experiment_config import FOLDER, N, FREQ, I, Q, SINGLE_SHOT
from qcore import Experiment, qua, Sweep
from qm import qua as qm_qua

number = "C"

qubit = f"q{number}"
resonator = f"rr{number}"
freq_sweeps = {
    "rrA": (200e6, 300e6),
    "rrB": (200e6, 300e6),
    "rrC": (-120e6, -60e6)
}
class SSreadout(Experiment):
    """ Determine highest allowed readout amp"""
    primary_datasets = ["I", "Q", "single_shot"]
    primary_sweeps = ["qubit_frequency"]
    def sequence(self):
        """QUA sequence that defines this Experiment subclass"""
        qua.update_frequency(self.qubit, self.qubit_frequency)
        self.qubit.play(self.qubit_pulse)
        self.resonator.play(self.test_readout_pulse, ampx=self.readout_amp)
        qua.align()
        self.resonator.measure(self.readout_pulse, (self.I, self.Q), ampx=self.ro_ampx, demod_type="dual")
        qm_qua.assign(self.single_shot,qm_qua.Cast.to_fixed(self.I > self.readout_pulse.threshold),)
        qua.wait(self.wait_time, self.resonator)
if __name__ == "__main__":
    """ """
    modes = {
        "qubit": qubit,
        "resonator": resonator,
    }
    pulses = {
        "qubit_pulse": f"{qubit}_gaussian_pi_pulse",
        "test_readout_pulse": f"{resonator}_test_readout_pulse",
        "readout_pulse": f"{resonator}_readout_pulse",
    }
    parameters = {
        "wait_time": 500_000,
        "ro_ampx": 1,
        "plot_single_shot": True,
    }
    N.num = 10000
    FREQ.name = "qubit_frequency"
    FREQ.start = freq_sweeps[resonator][0]
    FREQ.stop = freq_sweeps[resonator][1]
    FREQ.num = 101
    I_AMPX = Sweep(name="readout_amp", start=0.05, stop=1.0, step=0.05)
    sweeps = [N, I_AMPX, FREQ]
    Q.plot, I.plot = False, False
    SINGLE_SHOT.plot_args["plot_type"] = "image"
    SINGLE_SHOT.plot = True
    datasets = [I, Q, SINGLE_SHOT]
    expt = SSreadout(FOLDER, modes, pulses, sweeps, datasets, **parameters)
    expt.run()









