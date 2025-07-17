""" """
from config.experiment_config import FOLDER, N, FREQ, I, Q, SINGLE_SHOT
from qcore import Experiment, qua, Dataset
from qm import qua as qm_qua

# number = "
# qubit = f"q{number}"
# resonator = f"rr{number}"

# --------- To edit according to experiment ------------
qubit = "qubit"
resonator = "rr"
rect_decay_time = 1000
clear_decay_time = 40

I1 = Dataset(
    name="I1",
    save=False,
    plot=False,
)

I2 = Dataset(
    name="I2",
    save=False,
    plot=False,
)


Q1 = Dataset(
    name="Q1",
    save=False,
    plot=False,
)

Q2 = Dataset(
    name="Q2",
    save=False,
    plot=False,
)


SINGLE_SHOT_PRE = Dataset(
    name="single_shot_pre",
    save=True,
    plot=False,
)

SINGLE_SHOT_POST = Dataset(
    name="single_shot_post",
    save=True,
    plot=False,
)


class QNDConst(Experiment):
    """ Determine if readout pulse leads to quantum demolition"""

    primary_datasets = ["I1", "Q1", "I2", "Q2", "single_shot_pre", "single_shot_post"]

    def sequence(self):
        # Prepare qubit
        if self.excite_qubit:
          self.qubit.play(self.qubit_pulse)
          qua.align(self.resonator, self.qubit)
        
        # First measurement
        self.resonator.measure(self.readout_pulse, (self.I1, self.Q1), demod_type="dual")
       
        # Allow resonator to decay to not interfere with readout
        qua.wait(self.rect_decay_time, self.qubit, self.resonator)

        # Second measurement
        self.resonator.measure(self.readout_pulse, (self.I2, self.Q2), demod_type="dual")
        
        # Wait for reset
        qua.wait(self.wait_time, self.resonator)
        
        qm_qua.assign(self.single_shot_pre,qm_qua.Cast.to_fixed(self.I1 > self.readout_pulse.threshold),)
        qm_qua.assign(self.single_shot_post,qm_qua.Cast.to_fixed(self.I2 > self.readout_pulse.threshold),)

        # qm_qua.assign(self.non_demo,qm_qua.Cast.to_fixed(self.single_shot_post == self.single_shot_pre),)


class QNDClear(Experiment):
    """ Determine if readout pulse leads to quantum demolition"""

    primary_datasets = ["I1", "Q1", "I2", "Q2", "single_shot_pre", "single_shot_post"]
    
    def sequence(self):
        # Prepare qubit
        if self.excite_qubit:
          self.qubit.play(self.qubit_pulse)
          qua.align(self.resonator, self.qubit)
         
        # First measurement
        self.resonator.measure(self.clear_readout_pulse, (self.I1, self.Q1), demod_type="dual")

        # Allow resonator to decay to not interfere with readout
        qua.wait(self.clear_decay_time, self.qubit, self.resonator)

        # Second measurement
        self.resonator.measure(self.clear_readout_pulse, (self.I2, self.Q2), demod_type="dual")
        
        # Wait for reset
        qua.wait(self.wait_time, self.resonator)
        
        qm_qua.assign(self.single_shot_pre,qm_qua.Cast.to_fixed(self.I1 > self.clear_readout_pulse.threshold),)
        qm_qua.assign(self.single_shot_post,qm_qua.Cast.to_fixed(self.I2 > self.clear_readout_pulse.threshold),)

        # qm_qua.assign(self.non_demo,qm_qua.Cast.to_fixed(self.single_shot_post == self.single_shot_pre),)

if __name__ == "__main__":
    """ """
    modes = {
        "qubit": qubit,
        "resonator": resonator,
    }
    pulses = {
        "qubit_pulse": f"{qubit}_pi_8",
        "readout_pulse": f"{resonator}_readout_pulse",
        "clear_readout_pulse": f"{resonator}_CLEAR_readout_pulse",
    }
    parameters = {
        "rect_decay_time": rect_decay_time,
        "clear_decay_time": clear_decay_time,
        "wait_time": 500_000,
        "ro_ampx": 1,
        "excite_qubit": False
    }

    N.num = 20000
    FREQ.name = "repeat"
    FREQ.start = 1
    FREQ.stop = 2
    FREQ.num = 2
    FREQ.save = False

    sweeps = [N, FREQ]

    datasets = [I1, I2, Q1, Q2, SINGLE_SHOT_PRE, SINGLE_SHOT_POST]

    expt = QNDConst(FOLDER, modes, pulses, sweeps, datasets, **parameters)
    expt.run()
    
    expt = QNDClear(FOLDER, modes, pulses, sweeps, datasets, **parameters)
    expt.run()
    
    parameters["excite_qubit"] = True
    
    expt = QNDConst(FOLDER, modes, pulses, sweeps, datasets, **parameters)
    expt.run()
    
    expt = QNDClear(FOLDER, modes, pulses, sweeps, datasets, **parameters)
    expt.run()
    
    









