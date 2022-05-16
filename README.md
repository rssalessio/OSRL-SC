
# Code for Optimal Representation Learning in Multi-Task Bandits

**OSRL** (_Optimal Representation Learning in Multi-Task Bandits_) comprises an  algorithm that addresses the problem of _sample complexity with fixed confidence_ in Multi-Task Bandit problems.

The code contains not only the algorithm mentioned above, but also _KL-UCB_  [**1**], _D-Track and Stop_/_D-Track and Stop with challenger modification_ [**2**].

All the code has been written in Python or C.

## Hardware and Software setup

All experiments were executed on a stationary desktop computer,  featuring an Intel Xeon Silver 4110 CPU, 48GB of RAM. Ubuntu 18.04 was installed on the computer. Ubuntu is a open-source Operating System using the Linux kernel and based on Debian. For more  information, please check https://ubuntu.com/.

## Code and libraries

We set up our experiments using the following software and libraries:

* Python 3.7.7
* Cython version 0.29.15
* NumPy version 1.18.1
* SciPy version 1.4.1
* PyTorch version 1.4.0

All the code can be found in the folder `src`.

## Usage

You can run sample simulations by running the Jupyter notebooks located in the folder `notebooks`.

To run the notebooks you need to install Jupyter first. After that, you can open a shell in the `notebooks` directory and run

```bash
jupyter notebook
```

This will open the jupyter interface, where you can select which file to run.

## License

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.
Once the paper has been accepted, all the code will be published on GitHub with [MIT](https://choosealicense.com/licenses/mit/) license.

## References
[**1**] Garivier, Aur�lien, and Olivier Capp�. "The KL-UCB algorithm for bounded stochastic bandits and beyond." _Proceedings of the 24th annual conference on learning theory_. 2011.
[**2**] Garivier, Aur�lien, and Emilie Kaufmann. "Optimal best arm identification with fixed confidence." _Conference on Learning Theory_. 2016.
