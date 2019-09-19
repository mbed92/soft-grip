# Experiment description

Data in this folder contains of 54k train samples and 6k test samples. Each sample is a 12 numbers of readings from
acceleration with corresponding 1 number of a unified object's stiffness. Data is passed to the RNN in a batch of
200 consecutive data samples, what means that data is processed in a window with a length of 200 samples. Each batch
corresponds to one number of object's stiffness.

# Results (RNN + FC)

Results after XXX epochs shows that test error goes below 0.01 (which is ~0.45% of a maximum range of an estimated
parameter):