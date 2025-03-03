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
\mathrm{d} V(t) =&\ \mathbf{\tilde  F} X(t) \mathrm{d} t -\gamma V(t) \mathrm{d} t + \sqrt{\mathbf{\tilde L}} \mathrm{d} W(t).
\end{aligned}
$$

We can formulate the kinetic system as a special instance of the above OU process with $Z(t) = [X(t), V(t)]$ and

$$
\mathbf{F} = \begin{pmatrix}
0 & \mathbf{I}_d \\
\mathbf{\tilde  F} & -\gamma \mathbf{I}_d
\end{pmatrix} \text{ and } 
\mathbf{L} = \begin{pmatrix}
0 & 0 \\
0 & \mathbf{\tilde L}
\end{pmatrix}.
$$

Given the Gaussian initial distribution $Z(0)\sim p_0 = \mathcal{N}(m^x(0), \mathbf{P}^x_0) \times \mathcal{N}(m^v(0), \mathbf{P}^v_0)$, one can hence calculate the mean and covariance for $Z(t)$ using the above ODE system.

## Kinetic Fokker-Planck Equation with GMM potential
Consider the Kinetic Langevin dynamics

$$
\begin{aligned}
\mathrm{d} X(t) =&\ V(t) \mathrm{d} t, \\
\mathrm{d} V(t) =&\ -\nabla U(X(t)) \mathrm{d} t -\gamma V(t) \mathrm{d} t + \sqrt{\mathbf{\tilde L}} \mathrm{d} W(t).
\end{aligned}
$$

Consider the GMM potential consists of `n_Gaussian` Gaussians models. 
The covariance matrix of every Gaussian in the GMM is identity of size `domain_dim`$\times$`domain_dim`. 

## Kinetic McKean-Vlasov Equation with quadratic interaction
Consider the McKean-Vlasov equation where the interaction kernel $\Phi(x) = \frac{1}{2}|x|_\mathbf{A}^2$. Here $\mathbf{A}\in\mathbb{R}^{d\times d}$ is some constant positive definite matrix, and $|x|_\mathbf{A}$ denotes the $\ell^2$ norm of $x\in\mathbb{R}^d$, i.e. $|x|^2_\mathbf{A} = x^\top \mathbf{A} x$.
Consider the Kinetic interacting particle system
$$
\begin{aligned}
\mathrm{d} X(t) =&\ V(t) \mathrm{d} t, \\
\mathrm{d} V(t) =&\ -\nabla \Phi \ast \rho_t(X(t)) \mathrm{d} t -\gamma V(t) \mathrm{d} t + \sqrt{2} \mathrm{d} W(t),
\end{aligned}
$$
where $\rho_t$ denotes the marginal distribution of $X(t)$.
Under the assumption that $\int v f_0 \mathrm{d} x \mathrm{d} v = 0$ and $\mu_0 = 0$, it can be proved that the above interacting particle system is equivalent to 
$$
\begin{aligned}
\mathrm{d} X(t) =&\ V(t) \mathrm{d} t, \\
\mathrm{d} V(t) =&\ -\mathbf{A}X(t) \mathrm{d} t -\gamma V(t) \mathrm{d} t + \sqrt{2} \mathrm{d} W(t).
\end{aligned}
$$
Given the Gaussian initial distribution $Z(0)\sim p_0 = \mathcal{N}(m^x(0), \mathbf{P}^x_0) \times \mathcal{N}(m^v(0), \mathbf{P}^v_0)$, one can hence calculate the mean and covariance for $Z(t)$ using the above ODE system with
$$
\mathbf{F} = \begin{pmatrix}
0 & \mathbf{I}_d \\
-\mathbf{A} & -\gamma \mathbf{I}_d
\end{pmatrix} \text{ and } 
\mathbf{L} = \begin{pmatrix}
0 & 0 \\
0 & 2
\end{pmatrix}.
$$