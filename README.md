# Soft gripper 
Repository contains code and MuJoCo files for gathering data from MuJoCo simulation,
validate them and run a learning of a neural network in tensorflow. During prepared experiment the
soft gripper (based on the [MIT one](http://eprints.whiterose.ac.uk/95166/1/homberg2015haptic.pdf) and 
his 4-finger successor) catches an objects and perform exploratory moves (close and open).
Data from accelerometers attached to fingers is recorded. We experiment with such setup to see if there 
is any connection between sensor readings and physical parameters of an object. 

In order to run it you have to obtain your own license for MujoCo. See 
this [link](https://www.roboti.us/license.html).

## Description
### environment/
Scripts implemented in [mujoco_py](https://github.com/openai/mujoco-py) enable the 
user to interact with the environment.

### log/gripper/
XML files needed to run the model in MuJoCo and start the simulation.

### create_dataset.py
Script for generating and gathering data (logs data from accelerometers
 mounted on gripper's fingers) from simulation. During gathering a data a stiffness parameter
 of an object changes from 1e-5 to 1.5,
 
 ### dummy_regression.py
 Script with a neural network for a regression of a stiffness coefficient. Stiffness
 is defined as in the [main joint](http://www.mujoco.org/book/XMLreference.html#composite-joint)
 of a [composite/box](http://www.mujoco.org/book/XMLreference.html#composite)
 element of MuJoCo.
 
 ### playground.py
 Script for variety of things. Place to play with data.