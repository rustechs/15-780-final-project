# Dynamic System Configuration File

## System Dynamics

Abstractly, system dynamics are expressed in state-space form, with optional
additive gaussian noise (potentially time and state dependent) for both the process and measurement:

xDot = f(t,x) + N_p(mu_p(t,x),Sigma_p(t,x))
y = c(x) + N_m(mu_m(t,x),Sigma_m(t,x))

In the LTI case (with time and state invariant noise) this simplifies to:

xDot = Ax + N_p(mu_p,Sigma_p)
y = Cx + N_m(mu_m,Sigma_m)

## System Config Definitions
Systems are saved as MAT files to simplify parsing and reuse. 



