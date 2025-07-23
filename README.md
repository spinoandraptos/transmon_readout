# 🧪 Transmon Readout Characterisation Toolkit

This repository contains a suite of Python scripts for characterising and optimising the readout performance of superconducting transmon qubits. The workflow includes tools for pulse shaping, quantum non-demolition (QND) analysis, cavity decay extraction, and residual photon evaluation.

## 📁 Project Structure

```
transmon_readout/
├── Clear Optimisation/       # Generate optimal CLEAR pulses for fast, high-fidelity readout
├── Kappa Fit/                # Extract cavity decay rate κ from time-of-flight (TOF) experiments
├── QND Fit/                  # Quantify QND-ness of measurement using back-to-back measurements
├── Readout Power Fit/        # Determine max allowable readout power using Stark shift calibration
├── Residual Photon Fit/      # Estimate post-readout residual photon number
├── requirements_linux.txt    # Locked Python dependencies for Linux
├── requirements_windows.txt  # Locked Python dependencies for Windows
└── README.md                 # You are here
```

---

## 🛠️ Setup Instructions

This toolkit supports **both Windows and Linux**.

### 1. Clone the Repository

```bash
git clone https://github.com/spinoandraptos/transmon_readout.git
cd transmon_readout
git checkout production
```

### 2. Create a Virtual Environment (Recommended)

- **Windows:**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

- **Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

- **Windows:**

```powershell
pip install --upgrade pip
pip install -r requirements_windows.txt
```

- **Linux/macOS:**

```bash
pip install --upgrade pip
pip install -r requirements_linux.txt
```

---

## 💡 How to Use

Each folder contains a set of scripts or Jupyter notebooks specific to a task. Below is an overview:

### 🔵 `Clear Optimisation/`
- Generates an optimal CLEAR (Cavity Level Excitation and Reset) protocol pulse given system parameters.
- Goal: Maximise readout speed and fidelity by tuning multi-tone pulse parameters.
- **Key scripts:**
  - `ClearCalibrator.py` — Main script to run to optimize CLEAR pulse parameters given system parameters
  - `ClearOptimiser.py` — Helper script for ClearCalibrator, need not be modified typically
  - `EnvelopeSimulator.py` — Script for simulating the return envelope of a specified readout pulse
  - `ClearParamEditor.py` — Helper script that can be run to ensure CLEAR pulse timings and amplitude are legal after making manual segment-based edits
  - `ClearParamScaler.py` — Helper script that can be run to scale CLEAR pulse amplitude such that none lies above a specified threshold 
  - `ClearReadoutPulse.py` — Definition of CLEAR pulse in qcore
  - `EnvelopeSimulator.py` — Script for simulating the return envelope of a specified readout pulse
  - `EnvelopeSimulatorFormatter.py` — Script for converting the output CLEAR params from ClearCalibrator into a format that be directly copied-and-pasted into EnvelopeSimulator for visualisation
  - `{RR}_SystemParam.yml` — Configurations of system, requires the Kappa of Readout Resonator and Chi of Qubit-Resonator coupling, as well as a drive phase (which must be calibrated through actual measurement)

---

### 🟠 `Kappa Fit/`
- Fits exponential decay curves from time-domain cavity transmission data.
- Extracts the effective linewidth κ of the readout resonator.
- **Key scripts:**
  - `KappaFit.py` — Fits TOF hd5f measurement data to extract κ, given the starting time of the decay 

---

### 🟢 `QND Fit/`
- Performs conditional population tracking to assess QND-ness.
- Measures how much the measurement perturbs the qubit state.
- **Key scripts:**
  - `QND.py` — Calculates QND-ness from back-to-back measurements
  - `QNDExperiment.py` — Experiment setup for performing back-to-back single-shot measurements with qcore

---

### 🔴 `Readout Power Fit/`
- Uses the AC Stark shift of the qubit to determine safe readout power limits.
- Critical for avoiding measurement-induced qubit transitions.
- **Key scripts:**
  - `CalibratePower.py` — Fits Stark shift data to find maximum safe readout power.
  - `SSReadout.py` — Experiment setup for performing readout pulse stark shift measurements using varying readout power with qcore

---

### 🟣 `Residual Photon Fit/`
- Fits Ramsey fringe decay data to estimate leftover photons in the cavity post-readout.
- Useful for identifying the need for active cavity reset or pulse shaping.
- **Key scripts:**
  - `RamseyFit.py` — Fits Ramsey traces to known physical model to extract residual photon populations from measurements.
  - `ResidualPhotonExperiment.py` — Experiment setup for performing Ramsey (T2) experiments with varying delays post-pulse with qcore


---

## 📈 Example Workflow

1. Perform `TOF` experiment and save ADC trace

2. Extract κ using:

```bash
python "Kappa Fit/KappaFit.py"
```
3. Extract Chi using `RRSpecChi` experiment

4. Save Chi and Kappa in `Clear Optimisation/SystemParam.yml`
5. Tune readout power using:

```bash
python "Readout Power Fit/CalibratePower.py"
```
6. Update max allowed power in `Clear Optimisation/CalibrateClear.py`
7. Using a dummy pulse, or existing readout pulse, simulate the pulse envelope using `Clear Optimisation/EnvelopeSimulator.py`and compare with actual envelope obtained from measurement

8. Tune drive phase - typically one of [0.25, 0.75, 1.25, 1.75] - of `Clear Optimisation/SystemParam.yml` until simulation results resemble measurement envelope

9. Generate a CLEAR pulse with:

```bash
python "Clear Optimisation/CalibrateClear.py"
```

10. Evaluate residual photons:

```bash
python "Residual Photon Fit/ResidualPhotonExperiment.py"
python "Residual Photon Fit/RamseyFit.py"
```

11. Evaluate QND-ness:

```bash
python "QND Fit/QNDExperiment.py"
python "QND Fit/QND.py"
```

---

## 📚 Dependencies

The dependencies are pinned in the respective requirements files. Key packages include:

- `numpy`
- `scipy`
- `matplotlib`
- `cma`
- `h5py`
- `yaml`

---

## 📄 License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- Centre for Quantum Technologies (CQT), National University of Singapore  
- Quantum Circuits Research & Engineering Workgroup (QCREW)
- Inspired by established circuit QED readout techniques including:
  - CLEAR pulse shaping
  - AC Stark shift calibration
  - QND measurement fidelity estimation

---

## 🧑‍🔬 Author

**Juncheng Man**  
Research Intern [QCREW]  
Centre for Quantum Technologies, National University of Singapore  
📧 juncheng.man@u.nus.edu

---

*Feel free to raise issues or contribute improvements. Pull requests are welcome!*
