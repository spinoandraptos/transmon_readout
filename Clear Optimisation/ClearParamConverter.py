from dataclasses import dataclass

@dataclass
class DigitalWaveform:
    name: str

@dataclass
class ClearPulse:
    name: str
    I_ampx: float
    Q_ampx: float
    length: int
    pad: int
    ringdown1_amp: float
    ringup1_amp: float
    ringdown1_time: int
    ringup1_time: int
    ringdown2_amp: float
    ringdown2_time: int
    ringup2_amp: float
    ringup2_time: int
    drive_amp: float
    drive_time: int
    digital_marker: DigitalWaveform

# Input parameters
pulse = ClearPulse(
    name="rrB_CLEAR_readout_pulse",
    I_ampx=1.0,
    Q_ampx=0.0,
    length=952,
    pad=648,
    ringdown1_amp=0.01754149663818579,
    ringup1_amp=0.15586353877456202,
    ringdown1_time=100,
    ringup1_time=76,
    ringdown2_amp=-0.14260685048630622,
    ringdown2_time=69,
    ringup2_amp=0.04002529052650269,
    ringup2_time=83,
    drive_amp=0.0432,
    drive_time=624,
    digital_marker=DigitalWaveform("ADC_ON")
)

# === Scaling Factors ===
# These come from externally fitted results or optimization
rescaled_params = {
    "ringup1_amp": pulse.ringup1_amp,
    "ringdown1_amp": pulse.ringdown1_amp,
    "ringup2_amp": pulse.ringup2_amp,
    "drive_amp": pulse.drive_amp,
    "ringdown2_amp": pulse.ringdown2_amp,
}


# === New Optimal Times (in seconds) ===
optimal_times = {
    "optimal_ringup1_time": pulse.ringup1_time * 1e-9,
    "optimal_ringdown1_time": pulse.ringdown1_time * 1e-9,
    "optimal_ringup2_time": pulse.ringup2_time * 1e-9,
    "optimal_ringdown2_time": pulse.ringdown2_time * 1e-9,
    "pulse_start": 100e-9,
    "pulse_width": (pulse.ringup1_time + pulse.ringdown1_time + pulse.drive_time)* 1e-9
}

# === Print Final Output ===
print("\n# === Rescaled Parameters ===")
for k, v in rescaled_params.items():
    print(f"{k} = {v}")

print("\n# === Optimal Times (seconds) ===")
for k, v in optimal_times.items():
    print(f"{k} = {v}")
