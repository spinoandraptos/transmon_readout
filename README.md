# Transmon Readout

This repository provides a collection of Python notebooks to simulate and analyze the **transmon qubit readout** process via a **dispersively coupled resonator**. It includes:

- **ClearEndtoEnd.ipynb** aims to simulate the entire end to end process of a dispersive CLEAR readout by solving the Langevin equation for the resonator field in Heisenburg picture in both **rotating** and **lab** frames, and incorporates microwave signal processing emulating the experimental process 
- **RectEndtoEnd.ipynb** is the same as **CLEAREndtoEnd.ipynb** with the exception of using a rectangular readout pulse instead of CLEAR pulse
- **ClearSimulation.ipynb** aims to simulate the dynamics of the dispersively coupled system using the Lindblad Master Equation in QuTiP, and explores the effects of various types of collapse operators on system dymnamics and readout fidelity
- **MachineLearningCLEAR.ipynb** aims to optimise the CLEAR pulse shape and segment lengths using evolutionary algorithm to minimise both the stabilisation duration and reset duration of the resonator during ring-up and ring-down respectively
- **MachineLearningIntegration.ipynb** aims to optimise the window during which integration is applied to a reflected signal for the generation of I/Q distribution with the greatest separation and thus readout fidelity 
- **HeterodyneSimulation(2)HPC.py** are python scripts submitted to HPC clusters for the simulation of large number of quantum trajectories in system dynamics during heterodyne readout using the Stochastic Master Equation solver in QuTiP 
-**hpc_job.txt** is the job submitted to the PBS job scheduler of the HPC cluster
---

## Installation

```bash
git clone https://github.com/spinoandraptos/transmon_readout.git
cd transmon_readout
pip install -r requirements.txt
