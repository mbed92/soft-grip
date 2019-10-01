# Soft gripper 
Repository contains code and MuJoCo files for gathering data from MuJoCo simulation,
validate them and run a learning of a neural network in tensorflow. During prepared experiment the
soft gripper (based on the [MIT one](http://eprints.whiterose.ac.uk/95166/1/homberg2015haptic.pdf) and 
his 4-finger successor) catches an object and perform exploratory moves (tightening and loosening the embrace) - 
see the figure below. Data from accelerometers attached to fingers is recorded. We experiment with such a setup to see 
if there  is any connection between sensor readings and physical parameters of an object. 

![Gripper](https://github.com/mbed92/soft-grip/blob/master/images/overview.png?raw=true "Soft gripper")


In order to run it you have to obtain your own license for MujoCo. See [this link](https://www.roboti.us/license.html).

# TODO: README.md