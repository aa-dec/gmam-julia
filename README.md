# Implementation of the Geometric Minimum Action Method (GMAM) in Julia

See Hymann and vanden Eijnden 2008

Given the SDE
$$
dXt = drift(Xt) dt + sqrt{2 eps} sigma(X_t) dWt 
$$
and a stable equilibrium $x\_*$, 
the code computes 
- the instanton (maximum likelihood path) between $x\_*$ and $x$
- the quasipotential $V(x)$ at $x$
- the gradient of the quasipotential $\nabla V (x)$ at $x$
- the prefactor for the return time associated with the halfspace $D = \{ y | \nabla V(x) \cdot (y - x) \geq 0 \}$.

Work based loosely on 
 - Sharp Asymptotic Estimates for Expectations, Probabilities, and Mean First Passage Times in Stochastic Systems with Small Noise (https://arxiv.org/abs/2103.04837)
 - Generalisation of the Eyring-Kramers transition rate formula to irreversible diffusion processes (https://arxiv.org/abs/1507.02104)

## Compatibility Note

> [!WARNING]
> **This project strictly requires Julia v1.10 (LTS).** 
> 
> A core dependency of this solver is **Enzyme.jl**, which provides high-performance avant-garde Automatic Differentiation. Enzyme relies heavily on precise LLVM compiler structures and can be highly unstable or completely broken on newer Julia versions (e.g., 1.11+). 

To ensure stability, compatibility bounds are strictly locked to `julia = "1.10"` in the project configuration.