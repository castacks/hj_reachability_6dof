# HJ Reachability API Reference

## Overview

The `hj_reachability` package provides tools for solving Hamilton-Jacobi (HJ) Partial Differential Equations (PDEs) with a focus on reachability analysis in optimal control and differential games. The package is built on JAX for efficient numerical computation and GPU acceleration.

## Core Components

### Grid (`hj_reachability.grid`)

The `Grid` class represents the discretized state space where computations are performed.

```python
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    domain=hj.sets.Box(lo=np.array([...]), hi=np.array([...])),
    shape=(n1, n2, ..., nk),
    periodic_dims=None  # Optional: indices of periodic dimensions
)
```

Key attributes:
- `states`: JAX array containing all grid points
- `coordinate_vectors`: List of arrays containing coordinates along each dimension
- `domain`: The spatial domain of the grid
- `periodic_dims`: Indices of periodic dimensions (if any)

### Dynamics (`hj_reachability.dynamics`)

Base classes for defining system dynamics:

1. `ControlAndDisturbanceAffineDynamics`: Base class for systems with control and disturbance inputs that enter affinely.

```python
class MyDynamics(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, control_mode="min", disturbance_mode="max",
                 control_space=None, disturbance_space=None):
        super().__init__(control_mode, disturbance_mode, 
                        control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        """Return the drift term f(x,t)"""
        pass

    def control_jacobian(self, state, time):
        """Return the control input matrix g(x,t)"""
        pass

    def disturbance_jacobian(self, state, time):
        """Return the disturbance input matrix h(x,t)"""
        pass
```

### Solver (`hj_reachability.solver`)

Main functions for solving HJ PDEs:

1. `step(solver_settings, dynamics, grid, time, values, target_time)`:
   - Propagates the HJ PDE from current (time, values) to target_time
   - Returns the values at target_time

2. `solve(solver_settings, dynamics, grid, times, initial_values)`:
   - Solves the HJ PDE over a sequence of times
   - Returns array of values at each time step

```python
# Single step solution
target_values = hj.step(solver_settings, dynamics, grid, 
                       time=0.0, values=initial_values, 
                       target_time=-1.0)

# Multi-step solution
times = np.linspace(0, -2.0, 50)
all_values = hj.solve(solver_settings, dynamics, grid, 
                     times, initial_values)
```

### Solver Settings (`hj_reachability.solver.SolverSettings`)

Configuration for the numerical solver:

```python
settings = hj.SolverSettings.with_accuracy(
    accuracy="low"|"medium"|"high"|"very_high",
    hamiltonian_postprocessor=None  # Optional post-processing function
)
```

Accuracy levels affect:
- Spatial derivative approximations
- Time step sizes
- Artificial dissipation

### Sets (`hj_reachability.sets`)

Geometric primitives for defining domains and constraints:

1. `Box`: Rectangular domain
```python
box = hj.sets.Box(lo=np.array([x_min, y_min]), 
                  hi=np.array([x_max, y_max]))
```

2. `Ball`: Spherical domain
```python
ball = hj.sets.Ball(center=np.zeros(2), radius=1.0)
```

### Finite Differences (`hj_reachability.finite_differences`)

Numerical schemes for spatial derivatives:

1. `upwind_first`: First-order upwind schemes
   - Provides accurate derivative approximations for hyperbolic PDEs
   - Handles both periodic and non-periodic boundaries

## Example Usage

Here's a complete example solving a reachability problem for a simple system:

```python
import hj_reachability as hj
import numpy as np

# 1. Define system dynamics
class SimpleDynamics(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self):
        control_space = hj.sets.Box(np.array([-1.0]), np.array([1.0]))
        super().__init__("min", None, control_space, None)
    
    def open_loop_dynamics(self, state, time):
        return np.zeros_like(state)
    
    def control_jacobian(self, state, time):
        return np.eye(state.shape[-1])

# 2. Create computational grid
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
    (50, 50)
)

# 3. Define initial conditions
initial_values = np.linalg.norm(grid.states, axis=-1) - 1.0

# 4. Configure solver
solver_settings = hj.SolverSettings.with_accuracy("high")

# 5. Solve HJ PDE
times = np.linspace(0, -2.0, 41)
all_values = hj.solve(solver_settings, dynamics, grid, 
                     times, initial_values)
```

## Advanced Features

### Artificial Dissipation

The package includes artificial dissipation schemes to enhance numerical stability:

```python
from hj_reachability.artificial_dissipation import artificial_dissipation_from_accuracy
```

### Boundary Conditions

Support for various boundary conditions:
- Periodic boundaries
- Extrapolation
- Custom boundary handlers

### Time Integration

Time integration schemes available:
- Total Variation Diminishing Runge-Kutta schemes
- Custom time steppers

## Best Practices

1. **Grid Resolution**:
   - Start with coarse grids for quick testing
   - Refine grid resolution for production results
   - Consider periodic dimensions when appropriate

2. **Solver Settings**:
   - Use lower accuracy for prototyping
   - Increase accuracy for final results
   - Monitor convergence and stability

3. **Performance Optimization**:
   - Leverage JAX's JIT compilation
   - Use GPU acceleration for large problems
   - Profile and optimize bottlenecks

4. **Numerical Stability**:
   - Start with stable parameter values
   - Gradually increase complexity
   - Monitor for numerical artifacts

## Common Issues and Solutions

1. **Numerical Instability**:
   - Reduce time step size
   - Increase artificial dissipation
   - Check boundary conditions

2. **Performance Issues**:
   - Reduce grid resolution
   - Use JIT compilation
   - Consider GPU acceleration

3. **Boundary Effects**:
   - Extend computational domain
   - Adjust boundary conditions
   - Check for artificial reflections

## References

1. Mitchell, I. M. "The Flexible, Extensible and Efficient Toolbox of Level Set Methods"
2. Bansal, S. and Tomlin, C. "DeepReach: A Deep Learning Approach to High-Dimensional Reachability"
3. Osher, S. and Fedkiw, R. "Level Set Methods and Dynamic Implicit Surfaces"