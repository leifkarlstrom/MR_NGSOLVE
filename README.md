# Dynamics of viscoelastic medium surrounding an axisymmetric magma cavity #


This code in this repository provides a small python module
`magmaxisym` for simulating the dynamics of a Maxwell-type
viscoelastic medium surrounding an ellipsoidal axisymmetric magma
cavity. The module is entirely based on the open source finite element
library NGSolve.

* Prerequisite: Download and install NGSolve from ngsolve.org

* Class: The code provides a single python class `AxisymViscElas`
  which is defined in the file `axisymviscelas.py`.

* Driver: An example driver script can be found at `scripts/demo.py`. If you
  are new to this repo, start by opening a terminal, navigating to the
  `scripts` folder, and typing `netgen demo.py` into the terminal. To
  run without graphics, type `python3 demo.py` instead.

* Interactive use: Ensure `magmaxisym` is in your environment variable
  PYTHONPATH, then type in `import magmaxisym` into a python (or
  iPython) shell.


