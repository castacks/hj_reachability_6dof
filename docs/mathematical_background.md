# Mathematical Background: Hamilton-Jacobi Reachability Analysis

This document provides the mathematical foundations of Hamilton-Jacobi (HJ) reachability analysis and its implementation in the `hj_reachability` package.

## 1. System Dynamics

### 1.1 General Form

Consider a dynamical system:

```
dx/dt = f(x, u, d, t)
```

where:
- x ∈ ℝⁿ is the state
- u ∈ U is the control input
- d ∈ D is the disturbance input
- t ∈ [t₀, T] is time
- f: ℝⁿ × U × D × ℝ → ℝⁿ is the dynamics function

### 1.2 Control-Affine Systems

A common special case is control-affine systems:

```
dx/dt = f(x,t) + g(x,t)u + h(x,t)d
```

where:
- f(x,t) is the drift term
- g(x,t) is the control input matrix
- h(x,t) is the disturbance input matrix

## 2. Hamilton-Jacobi-Isaacs PDE

### 2.1 Value Function

The value function V(x,t) represents the optimal cost-to-go:

```
V(x,t) = min_u max_d J(x,t,u(·),d(·))
```

where J is the cost functional.

### 2.2 HJI PDE

The value function satisfies the Hamilton-Jacobi-Isaacs PDE:

```
∂V/∂t + H(x, ∇V, t) = 0
```

with terminal condition:
```
V(x,T) = l(x)
```

where:
- H is the Hamiltonian
- l(x) is the terminal cost function
- ∇V is the spatial gradient of V

### 2.3 Hamiltonian

For control-affine systems:

```
H(x,p,t) = p^T f(x,t) + min_u p^T g(x,t)u + max_d p^T h(x,t)d
```

where p = ∇V is the costate.

## 3. Numerical Methods

### 3.1 Spatial Discretization

The package uses finite difference methods for spatial derivatives:

1. First-order upwind schemes:
```
(∂V/∂xᵢ)⁺ ≈ (V_{i+1} - V_i)/Δx
(∂V/∂xᵢ)⁻ ≈ (V_i - V_{i-1})/Δx
```

2. Artificial dissipation for stability:
```
∂²V/∂x² ≈ (V_{i+1} - 2V_i + V_{i-1})/Δx²
```

### 3.2 Time Integration

Total Variation Diminishing Runge-Kutta schemes:

Third-order TVD-RK3:
```
V¹ = Vⁿ + Δt L(Vⁿ)
V² = ¾Vⁿ + ¼V¹ + ¼Δt L(V¹)
Vⁿ⁺¹ = ⅓Vⁿ + ⅔V² + ⅔Δt L(V²)
```

where L is the spatial operator.

### 3.3 CFL Condition

For numerical stability:
```
Δt ≤ CFL * min(Δx/max|f|)
```

where CFL < 1 is the Courant number.

## 4. Reachability Analysis

### 4.1 Backward Reachable Set (BRS)

The set of initial states that can reach a target set:

```
BRS(t) = {x₀ | ∃u(·),∀d(·), x(T) ∈ Target}
```

Computed as the zero sublevel set:
```
BRS(t) = {x | V(x,t) ≤ 0}
```

### 4.2 Backward Reachable Tube (BRT)

The set of states that can reach the target at any time:

```
BRT(t) = {x₀ | ∃u(·),∀d(·),∃τ∈[t,T], x(τ) ∈ Target}
```

### 4.3 Level Set Method

The package uses the level set method where:
1. Target set is represented as zero sublevel set of l(x)
2. Value function evolves according to HJI PDE
3. Reachable set is extracted as zero sublevel set

## 5. Implementation Details

### 5.1 Grid Structure

The package uses a rectangular grid:
```python
grid = Grid.from_lattice_parameters_and_boundary_conditions(
    domain=Box(lo, hi),
    shape=(n₁, n₂, ..., nₖ),
    periodic_dims=None
)
```

### 5.2 Numerical Schemes

1. Spatial derivatives:
```python
from hj_reachability.finite_differences import upwind_first
derivatives = upwind_first(grid, values)
```

2. Time integration:
```python
from hj_reachability.time_integration import step
next_values = step(dynamics, grid, values, time_step)
```

### 5.3 Boundary Conditions

1. Periodic:
```
V(x + L) = V(x)
```

2. Extrapolation:
```
∂V/∂n = 0
```

where n is the outward normal.

## 6. Practical Considerations

### 6.1 Grid Resolution

Trade-off between accuracy and computation:
- Coarse grid: Fast but inaccurate
- Fine grid: Accurate but slow
- Rule of thumb: Start coarse, refine until convergence

### 6.2 Time Step Selection

1. CFL condition for stability
2. Accuracy requirements
3. Computational budget

### 6.3 Numerical Stability

1. Use appropriate artificial dissipation
2. Monitor energy/mass conservation
3. Check for oscillations

## 7. Advanced Topics

### 7.1 Decomposition Methods

For high-dimensional systems:
1. Projections
2. Tensor decomposition
3. Neural network approximations

### 7.2 Optimal Control Synthesis

1. Extract optimal control:
```
u*(x) = argmin_u p^T g(x)u
```

2. Implement feedback law:
```
u(t) = u*(x(t))
```

### 7.3 Robustness Analysis

1. Disturbance as uncertainty:
```
d ∈ D = {d | ||d|| ≤ r}
```

2. Robust reachable sets:
```
BRS_r(t) = {x₀ | ∃u(·),∀d(·)∈D, x(T) ∈ Target}
```

## References

1. Evans, L.C. "Partial Differential Equations"
2. Mitchell, I.M. "A Toolbox of Level Set Methods"
3. Osher, S. and Fedkiw, R. "Level Set Methods and Dynamic Implicit Surfaces"
4. Bansal, S. et al. "Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances"