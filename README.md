# orbital-mechanics
A series of Python scripts used to solve exercises of the Orbital Mechanics course at ETSIAE using the NumPy package. Each script contains functions that solve the proposed problems in an efficient and simple way.

The first exercise includes 2 functions that output all the required parameters and arrays. Initial arrays of position and velocity must be created. The output tuple can be asigned and sliced to obtain specific values. A previous study of the output variables of each function is required before slicing.

The second exercise solves the problem using two functions. The first function determines the trajectory parameters and the type of orbit using the Newton-Raphson method to determine the semi-mayor axis of the orbit. It ensures quick convergence while using the benefits of NumPy. The analytical expressions of the derivatives were calculated by hand and simplified for better readibility. The second function calculates the initial and arrival velocities, as well as other parameters.
