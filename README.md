# PDE-inverse-problem

# OU process
Consider the OU process with a Gaussian initial distribution $Z(0) \sim p_0 = \mathcal{N}(m_0, \mathbf{P}_0)$,

$$
\mathrm{d} Z(t) = \mathbf{F} Z(t) \mathrm{d} t + \sqrt{\mathbf{L}} \mathrm{d} W(t)
$$

We know that $Z(t)$ has the distribution $p_t = \mathcal{N}(m_t, \mathbf{P}_t)$, where the mean $m(t)$ and covariance $\mathbf{P}_t$ evolve according to the following ODE

$$
\mathrm{d} m(t) = \mathbf{F} m(t) \mathrm{d} t, \quad \mathrm{d} \mathbf{P}_t = \mathbf{F} \mathbf{P}_t + \mathbf{P}_t\mathbf{F}^\top + \mathbf{L}.
$$

## Kinetic OU process
Consider the Kinetic Langevin dynamics

$$
\begin{aligned}
\mathrm{d} X(t) =&\ V(t) \mathrm{d} t, \\
\mathrm{d} V(t) =&\ \mathbf{\tilde  F} X(t) \mathrm{d} t + \sqrt{\mathbf{\tilde L}} \mathrm{d} W(t).
\end{aligned}
$$

We can formulate the kinetic system as a special instance of the above OU process with $Z(t) = [X(t), V(t)]$ and

$$
\mathbf{F} = \begin{pmatrix}
0 & \mathbf{I}_d \\
\mathbf{\tilde  F} & 0
\end{pmatrix} \text{ and } 
\mathbf{L} = \begin{pmatrix}
0 & 0 \\
0 & \mathbf{\tilde L}
\end{pmatrix}.
$$

Given the Gaussian initial distribution $Z(0)\sim p_0 = \mathcal{N}(m^x(0), \mathbf{P}^x_0) \times \mathcal{N}(m^v(0), \mathbf{P}^v_0)$, one can hence calculate the mean and covariance for $Z(t)$ using the above ODE system.

## Kinetic Fokker-Planck Equation
