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

Each folder contains a set of scripts specific to a task. Below is an overview:

### 🔵 `Clear Optimisation/`
- Generates an optimal CLEAR (Cavity Level Excitation and Reset) protocol pulse given system parameters.
- Goal: Maximise readout speed and fidelity by optimising multi-segment pulse parameters.
- **Key scripts:**
  - `ClearFormatter.py` — Helper script for parsing pulse params using qcore syntax for simulation scripts [do not modify unless needed]
  - `ClearOptimiser.py` — Main script to run to optimize CLEAR pulse parameters given system parameters
  - `ClearReadoutPulse.py` — Definition of CLEAR pulse in qcore
  - `EnvelopeSimulator.py` — Script for simulating the return envelope, photon number, and phase-space trajectories of a readout pulse
  - `ReadoutSimulator.py` — Helper script that simulates the dynamics of dispersive readout [do not modify unless needed]
  - `SysParamFinder.py` — Script for fitting the system parameters needed for high-fidelity simulation using reference envelopes from actual measurement
  - `{RR}_SystemParam.yml` — Configurations of system, requires system parameters fitted through `SysParamFinder.py`

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

1. Extract Chi using `RRSpecChi` experiment
2. Save Chi in `Clear Optimisation/SystemParam.yml`
3. Choose a dummy/exisiting pulse and extract return envelope trace for both |g> and |e> using `IntegrationWeightsTraining` experiment
4. Save pulse params and envelope traces, and fit system params using:
```bash
python "Clear Optimisation/SysParamFinder.py"
```
6. Update `Clear Optimisation/SystemParam.yml` with the set of params found from fitting results
7. Generate a CLEAR pulse with:
```bash
python "Clear Optimisation/ClearOptimiser.py"
```
8. Simulate return envelope, photon number, and phase-space trajectories of the CLEAR pulse using:
```bash
python "Clear Optimisation/EnvelopeSimulator.py"
```
9. If CLEAR pulse is not good, modify the params search range and cost weights in `Clear Optimisation/ClearOptimiser.py` and repeat the above.
10. Evaluate readout fidelity using `ThresholdCalculation` experiment
11. Evaluate residual photons to verify clearing:

```bash
python "Residual Photon Fit/ResidualPhotonExperiment.py"
python "Residual Photon Fit/RamseyFit.py"
```

12. Evaluate QND-ness:

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
- `optuna`
- `skopt`
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
