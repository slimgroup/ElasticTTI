# Elastic TTI

This repository contains a simple implementation of the 2D elastic TTI wave equation. This is fully implemented with [Devito](https://github.com/devitocodes/devito). This is an experimental setup and only provides basic utilities needed for elastic TTI modeling. These basic utilities are:

- `SeismicStiffness`: That creates the 4th order elastic tensor from the TTI parameters `epsilon, delta, theta, phi, rho`. The additional `gamma` parameter isn't yet implemented.  This implements the 3D elastic stiffness tensor as well. The implementation of the transformatin and tensor operation are not currently designed to be optimal but to be mathematically elegant and can therefore take some time to initialize the Stiffness tensor.

- `divrot, gradrot`: custom divergence and gradient operator using a combination of standard FD and 45 degree rotated FD for better accuracy and less dispersion.

## Example

We provide a single simple example in `examples/tti_modeling.py` that can be run from the main directory as:

```bash
PYTHONPATH=$PWD python3 examples/tti_modeling.py
```

and will plot the wavefield after the simulation. This implements a simple two layer model.

![wavefields](https://raw.githubusercontent.com/slimgroup/ElasticTTI/master/examples/2DETTI.png)
: Elastic TTI wavefields

## Author

Mathias Louboutin<mlouboutin3@gatech.edu>
