# HJ Reachability Tutorial

This tutorial walks through a practical example of using the `hj_reachability` package to solve a reachability problem for a simple vehicle model.

## Prerequisites

Before starting, make sure you have the following installed:
```bash
pip install --upgrade hj-reachability jax numpy matplotlib
```

## Problem Setup: Vehicle Reachability Analysis

We'll analyze a simple vehicle model with the following state variables:
- (x, y): Position in 2D plane
- v: Velocity
- θ: Heading angle

The control inputs are:
- a: Acceleration
- ω: Angular velocity (steering)

The goal is to compute the backward reachable set: the set of initial states from which the vehicle can reach a target set within a given time horizon.

## Step 1: Define System Dynamics

First, we'll define the vehicle dynamics using the `ControlAndDisturbanceAffineDynamics` class:

```python
import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

class VehicleDynamics(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, max_acceleration=1.0, max_steering=1.0):
        # Define control space as a box: [acceleration, steering]
        control_space = hj.sets.Box(
            lo=jnp.array([-max_acceleration, -max_steering]),
            hi=jnp.array([max_acceleration, max_steering])
        )
        super().__init__(control_mode="min", 
                        disturbance_mode=None,
                        control_space=control_space,
                        disturbance_space=None)

    def open_loop_dynamics(self, state, time):
        """Drift term f(x,t)"""
        x, y, v, theta = state
        return jnp.array([
            v * jnp.cos(theta),  # dx/dt
            v * jnp.sin(theta),  # dy/dt
            0.0,                 # dv/dt
            0.0                  # dθ/dt
        ])

    def control_jacobian(self, state, time):
        """Control input matrix g(x,t)"""
        x, y, v, theta = state
        return jnp.array([
            [0.0, 0.0],    # Effect on dx/dt
            [0.0, 0.0],    # Effect on dy/dt
            [1.0, 0.0],    # Effect on dv/dt
            [0.0, 1.0]     # Effect on dθ/dt
        ])
```

## Step 2: Create Computational Grid

Next, we'll create a grid that covers our state space:

```python
# Define grid bounds and resolution
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    domain=hj.sets.Box(
        lo=np.array([-5.0, -5.0, -2.0, -np.pi]),  # [x_min, y_min, v_min, θ_min]
        hi=np.array([5.0, 5.0, 2.0, np.pi])       # [x_max, y_max, v_max, θ_max]
    ),
    shape=(41, 41, 20, 25),  # Grid resolution in each dimension
    periodic_dims=3          # θ is periodic
)
```

## Step 3: Define Initial Value Function

The initial value function defines our target set. Here, we'll define it as a circle in the x-y plane:

```python
def create_initial_values(grid, target_radius=1.0):
    # Distance from origin in x-y plane
    xy_distance = jnp.linalg.norm(grid.states[..., :2], axis=-1)
    
    # Negative inside target set, positive outside
    return xy_distance - target_radius
```

## Step 4: Configure Solver and Solve

Now we can set up the solver and compute the backward reachable set:

```python
# Create dynamics instance
dynamics = VehicleDynamics(max_acceleration=1.0, max_steering=1.0)

# Initial conditions
initial_values = create_initial_values(grid)

# Solver settings
solver_settings = hj.SolverSettings.with_accuracy(
    "high",
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

# Solve backward in time
times = np.linspace(0, -2.0, 41)  # 2 second backward horizon
all_values = hj.solve(solver_settings, dynamics, grid, 
                     times, initial_values)
```

## Step 5: Visualize Results

We can visualize the results using matplotlib:

```python
import matplotlib.pyplot as plt

def plot_slice(grid, values, v_idx=10, theta_idx=12):
    """Plot x-y slice at fixed v and θ."""
    plt.figure(figsize=(10, 8))
    
    # Create contour plot
    plt.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[..., v_idx, theta_idx].T,
        levels=[0],
        colors='black',
        linewidths=2
    )
    
    plt.fill_between([-5, 5], [-5, -5], [5, 5], 
                    alpha=0.1, color='blue')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Backward Reachable Set\n'
             f'v = {grid.coordinate_vectors[2][v_idx]:.2f}, '
             f'θ = {grid.coordinate_vectors[3][theta_idx]:.2f}')
    plt.axis('equal')
    plt.show()

# Plot initial and final sets
plot_slice(grid, initial_values)
plot_slice(grid, all_values[-1])
```

## Understanding the Results

The zero level set of the value function represents the boundary of the reachable set:
- Points inside the zero level set (negative values) can reach the target
- Points outside (positive values) cannot reach the target

The backward reachable set shows all initial states from which the vehicle can reach the target circle within 2 seconds.

## Advanced Topics

### 1. Adding Disturbances

To model uncertainty, we can add bounded disturbances to the dynamics:

```python
def __init__(self, max_acceleration=1.0, max_steering=1.0, 
             max_disturbance=0.1):
    control_space = hj.sets.Box(
        lo=jnp.array([-max_acceleration, -max_steering]),
        hi=jnp.array([max_acceleration, max_steering])
    )
    disturbance_space = hj.sets.Ball(
        center=jnp.zeros(2),
        radius=max_disturbance
    )
    super().__init__("min", "max", control_space, disturbance_space)

def disturbance_jacobian(self, state, time):
    """Disturbance input matrix h(x,t)"""
    return jnp.array([
        [1.0, 0.0],  # Disturbance affects x
        [0.0, 1.0],  # Disturbance affects y
        [0.0, 0.0],  # No effect on v
        [0.0, 0.0]   # No effect on θ
    ])
```

### 2. Performance Optimization

For better performance:

1. Use JIT compilation:
```python
from jax import jit

solve_jit = jit(lambda t, v: hj.step(
    solver_settings, dynamics, grid, 0.0, v, t
))
```

2. Adjust grid resolution based on needs:
- Coarse grid for prototyping
- Fine grid for final results

3. Use GPU acceleration when available:
```python
import jax
jax.config.update('jax_platform_name', 'gpu')
```

## Common Pitfalls

1. **Grid Resolution**
   - Too coarse: Inaccurate results
   - Too fine: Slow computation
   - Solution: Start coarse, refine as needed

2. **Numerical Issues**
   - Check CFL condition
   - Use appropriate artificial dissipation
   - Monitor for instabilities

3. **Boundary Effects**
   - Make domain large enough
   - Handle periodic dimensions correctly
   - Check for artificial boundary effects

## Next Steps

1. Experiment with different:
   - Target sets
   - Time horizons
   - System dynamics
   - Control constraints

2. Advanced applications:
   - Safety analysis
   - Path planning
   - Robust control synthesis

## References

For more information:
1. Level Set Methods: Osher & Fedkiw
2. Reachability Analysis: Mitchell's Toolbox
3. JAX Documentation: https://jax.readthedocs.io