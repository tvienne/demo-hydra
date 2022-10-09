# Demo Hydra for machine learning

Thie repo contains the same application (machine learning model training) with different
ways to handle configuration files.


*Step 1 - launch main_no_conf.py* : which trains a vanilla neural 
network with no configuration file. All variables are defined in the script.

*Step 2 - launch main_with_argparse.py* : the parameters of the neural network are now handled with
argparse.

*Step 3 - launch main_with_conf.py* : the parameters of the neural network are now handled with
a configuration file.

*Step 4 - launch main_with_hydra.py* : the parameters of the neural network are now handled with
the Hydra framework.