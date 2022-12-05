# Data Fusion by HDM and ML

The current repository contains the program code allowing one to construct the absolute permeability map consistent with the interpreted results of well logging and well test measurements in oil reservoirs. The two-dimensional spatial distribution of the rock permeability is approximated by the Nadaraya-Watson kernel regression. Parameters of the kernel regression are tuned by solving the optimization problem in which, for each well placed in an oil reservoir, the differences between the actual and predicted values of absolute permeability at the well location, absolute integral permeability of the domain around the well, and skin factor are minimized. The inverse problem is solved via multiple solutions to forward problems, in which we estimate the integral permeability of the reservoir surrounding a well and the skin factor by an artificial neural network. The latter is trained on the physics-based synthetic dataset generated using the procedure comprising the numerical simulation of the bottomhole pressure decline curve in the reservoir simulator followed by its interpretation using a semi-analytical reservoir model. The developed method for reservoir permeability map construction is applied to the Egg Model containing highly heterogeneous permeability distribution due to the presence of highly-permeable channels.


We demonstrate the code capabilities in the notebook **'main_notebook.ipynb'**. 

The program code is located in the folder **'src'**. 

Various datasets and saved ANNs can be found in **'data'** folder.

Hydrodynamical models for MUFITS simulator are saved in **'MUFITS_models'** folder.


