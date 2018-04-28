# Dynamic System Configuration File

## System Dynamics

Abstractly, system dynamics are expressed in state-space form, with 
additive gaussian noise in the measurement:

xDot = f(t,x)
y = c(x) + N(mu(t,x),Sigma(t,x))

In the LTI case (with time and state invariant noise) this simplifies to:

xDot = Ax
y = Cx + N(mu,Sigma)

Naturally, measurements are taken at discrete time intervals so the actual measurement data is given by:

y_meas = Sample(y,T_sample)

Where T_sample is the sampling period of the given measurement system (assumed homogeneous across all measured states).

## System Config Definitions
The various variables/functions involved in the definitions of the dynamics described above are saved in self-contained MAT files for ease of reuse.