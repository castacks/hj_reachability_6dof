import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

class Quadrotor6DOF(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    A 6-DOF quadrotor model using differentially flat outputs.
    
    State variables:
    - x, y, z: Position in world frame
    - vx, vy, vz: Velocities in world frame
    - psi: Yaw angle
    - psi_dot: Yaw rate
    
    Control inputs (normalized between -1 and 1):
    - ax: Desired acceleration in x
    - ay: Desired acceleration in y
    - az: Desired acceleration in z (additional to gravity)
    - psi_ddot: Desired yaw acceleration
    
    The model assumes perfect tracking of desired accelerations through differential flatness,
    which is a reasonable assumption for a well-tuned quadrotor with high-bandwidth control.
    """

    def __init__(self,
                 max_velocity=5.0,  # m/s
                 max_acceleration=5.0,  # m/s^2
                 max_yaw_rate=3.14,  # rad/s
                 max_yaw_accel=3.14,  # rad/s^2
                 control_mode="min",
                 disturbance_mode="min",  # Changed to "min" to match control mode
                 control_space=None,
                 disturbance_space=None):
        """
        Initialize the quadrotor model.
        
        Args:
            max_velocity: Maximum velocity in any direction (m/s)
            max_acceleration: Maximum acceleration in any direction (m/s^2)
            max_yaw_rate: Maximum yaw rate (rad/s)
            max_yaw_accel: Maximum yaw acceleration (rad/s^2)
            control_mode: "min" or "max" for optimal control objective
            disturbance_mode: "min" or "max" for disturbance behavior
            control_space: Optional custom control space
            disturbance_space: Optional custom disturbance space
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_yaw_rate = max_yaw_rate
        self.max_yaw_accel = max_yaw_accel
        
        # Define control space if not provided
        if control_space is None:
            control_space = sets.Box(
                lo=jnp.array([-max_acceleration, -max_acceleration, 
                             -max_acceleration, -max_yaw_accel]),
                hi=jnp.array([max_acceleration, max_acceleration, 
                             max_acceleration, max_yaw_accel])
            )
        
        # Define a minimal disturbance space (required by the solver)
        if disturbance_space is None:
            disturbance_space = sets.Box(
                lo=jnp.array([-0.1]),  # Small disturbance
                hi=jnp.array([0.1])
            )
        
        super().__init__(control_mode, disturbance_mode, 
                        control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        """
        Compute the drift term f(x,t) of the dynamics.
        
        The drift term represents the natural evolution of the state
        when no control input is applied.
        """
        # Unpack state
        _, _, _, vx, vy, vz, _, psi_dot = state
        
        # Natural dynamics (velocity affects position, gravity affects z)
        return jnp.array([
            vx,              # dx/dt = vx
            vy,              # dy/dt = vy
            vz,              # dz/dt = vz
            0.0,             # dvx/dt = 0 (no drift in acceleration)
            0.0,             # dvy/dt = 0
            -9.81,           # dvz/dt = -g (gravity)
            psi_dot,         # dpsi/dt = psi_dot
            0.0              # dpsi_dot/dt = 0
        ])

    def control_jacobian(self, state, time):
        """
        Compute the control input matrix g(x,t).
        
        This matrix maps control inputs to state derivatives.
        """
        # Control inputs directly affect accelerations and yaw acceleration
        return jnp.array([
            [0.0, 0.0, 0.0, 0.0],    # x
            [0.0, 0.0, 0.0, 0.0],    # y
            [0.0, 0.0, 0.0, 0.0],    # z
            [1.0, 0.0, 0.0, 0.0],    # vx
            [0.0, 1.0, 0.0, 0.0],    # vy
            [0.0, 0.0, 1.0, 0.0],    # vz
            [0.0, 0.0, 0.0, 0.0],    # psi
            [0.0, 0.0, 0.0, 1.0]     # psi_dot
        ])

    def disturbance_jacobian(self, state, time):
        """
        No disturbance model implemented.
        Override this method if disturbance needs to be added.
        """
        return jnp.zeros((8, 1))  # Placeholder for no disturbance

    def get_optimal_control(self, state, costate):
        """
        Compute optimal control given current state and costate.
        
        This is useful for extracting the optimal control policy
        from the value function gradient (costate).
        """
        # Extract relevant components of costate
        px_v = costate[3:6]  # Costate for velocities
        ppsi_v = costate[7]  # Costate for yaw rate
        
        # Control authority matrix from control_jacobian
        g = self.control_jacobian(state, 0.0)
        
        # Compute optimal control based on Pontryagin's Maximum Principle
        if self.control_mode == "min":
            optimal_control = -self.control_space.project(
                jnp.array([px_v[0], px_v[1], px_v[2], ppsi_v]))
        else:  # max
            optimal_control = self.control_space.project(
                jnp.array([px_v[0], px_v[1], px_v[2], ppsi_v]))
        
        return optimal_control