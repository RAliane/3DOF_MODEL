"""
Module for implementing a neural network (Multi-Layer Perceptron)
that backpropagates through environmental, rigid body dynamics,
trajectory, and wear-and-tear models.
"""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.datasets import make_regression

class Vector:
    """
    A utility class for creating and handling 3D vectors
    (position, velocity, acceleration) in a North-East-Down (NED) coordinate system.
    """

    @staticmethod
    def position_vector(north: float, east: float, down: float) -> np.ndarray:
        """
        Creates a position vector in NED coordinates.
        """
        return np.array([north, east, down])

    @staticmethod
    def velocity_vector(v_north: float, v_east: float, v_down: float) -> np.ndarray:
        """
        Creates a velocity vector in NED coordinates.
        """
        return np.array([v_north, v_east, v_down])

    @staticmethod
    def acceleration_vector(a_north: float, a_east: float, a_down: float) -> np.ndarray:
        """
        Creates an acceleration vector in NED coordinates.
        """
        return np.array([a_north, a_east, a_down])

    @staticmethod
    def magnitude(vector: np.ndarray) -> float:
        """
        Calculates the magnitude of a 3D vector.
        """
        return np.linalg.norm(vector)

class Environment:
    """
    Models environmental parameters, including dynamic pressure,
    using ISA (International Standard Atmosphere) for troposphere and stratosphere.
    """
    g = 9.80665  # Gravitational acceleration (m/s²)
    R = 287.05   # Specific gas constant for air (J/(kg·K))

    # Sea level conditions
    T0 = 288.15  # Temperature (K)
    P0 = 101325  # Pressure (Pa)
    rho0 = 1.225 # Density (kg/m³)

    # Tropopause (11 km)
    h_tropopause = 11000  # Altitude (m)
    T11 = 216.65         # Temperature at tropopause (K)
    P11 = 22632          # Pressure at tropopause (Pa)

    # Lapse rate (K/m)
    L = -0.0065

    @staticmethod
    def _temperature(h: float) -> float:
        """
        Calculates temperature (K) at altitude h (m).
        """
        if h <= Environment.h_tropopause:
            return Environment.T0 + Environment.L * h
        else:
            return Environment.T11

    @staticmethod
    def _pressure(h: float) -> float:
        """
        Calculates pressure (Pa) at altitude h (m).
        """
        if h <= Environment.h_tropopause:
            return Environment.P0 * (1 + (Environment.L * h) / Environment.T0) ** (-Environment.g / (Environment.R * Environment.L))
        else:
            return Environment.P11 * math.exp(-Environment.g * (h - Environment.h_tropopause) / (Environment.R * Environment.T11))

    @staticmethod
    def density(h: float) -> float:
        """
        Calculates air density (kg/m³) at altitude h (m).
        """
        T = Environment._temperature(h)
        P = Environment._pressure(h)
        return P / (Environment.R * T)

    @staticmethod
    def dynamic_pressure(position: np.ndarray, velocity: np.ndarray) -> float:
        """
        Calculates dynamic pressure (Pa) at a given position (NED) and velocity (NED).
        """
        h = -position[2]  # Altitude is the negative of the 'down' component in NED
        velocity_magnitude = Vector.magnitude(velocity)
        rho = Environment.density(h)
        return 0.5 * rho * velocity_magnitude**2

class Aerodynamics:
    """Models aerodynamic forces (lift, drag, thrust) in NED coordinates."""

    def __init__(
        self,
        wing_area: float,
        aspect_ratio: float,
        oswald_efficiency: float,
        Cd0: float,  # Zero-lift drag coefficient
        Cl: float,   # Lift coefficient
        propulsion: Propulsion,
    ):
        self.wing_area = wing_area
        self.aspect_ratio = aspect_ratio
        self.oswald_efficiency = oswald_efficiency
        self.Cd0 = Cd0
        self.Cl = Cl
        self.propulsion = propulsion

    def dynamic_pressure(self, velocity: np.ndarray, position: np.ndarray, environment: Environment) -> float:
        """Calculates dynamic pressure (q) at the given position and velocity."""
        h = -position[2]  # Altitude from NED 'down'
        rho = environment.density(h)
        v_mag = Vector.magnitude(velocity)
        return 0.5 * rho * v_mag**2

    def lift_force(self, velocity: np.ndarray, position: np.ndarray, environment: Environment) -> np.ndarray:
        """Calculates lift force vector (NED)."""
        q = self.dynamic_pressure(velocity, position, environment)
        lift_magnitude = q * self.Cl * self.wing_area

        # Simplified: Lift is perpendicular to velocity (in the N-E plane)
        vel_unit = Vector.unit_vector(velocity)
        lift_direction = np.array([-vel_unit[1], vel_unit[0], 0.0])  # Perpendicular in N-E plane
        lift_direction = lift_direction / Vector.magnitude(lift_direction)
        return lift_magnitude * lift_direction

    def parasitic_drag_force(self, velocity: np.ndarray, position: np.ndarray, environment: Environment) -> np.ndarray:
        """Calculates parasitic drag force vector (NED)."""
        q = self.dynamic_pressure(velocity, position, environment)
        drag_magnitude = q * self.Cd0 * self.wing_area
        drag_direction = -Vector.unit_vector(velocity)  # Opposes velocity
        return drag_magnitude * drag_direction

    def induced_drag_force(self, velocity: np.ndarray, position: np.ndarray, environment: Environment) -> np.ndarray:
        """Calculates induced drag force vector (NED)."""
        q = self.dynamic_pressure(velocity, position, environment)
        lift_magnitude = q * self.Cl * self.wing_area
        induced_drag_magnitude = (lift_magnitude**2) / (q * self.wing_area * np.pi * self.oswald_efficiency * self.aspect_ratio)
        drag_direction = -Vector.unit_vector(velocity)  # Opposes velocity
        return induced_drag_magnitude * drag_direction

    def total_drag_force(self, velocity: np.ndarray, position: np.ndarray, environment: Environment) -> np.ndarray:
        """Calculates total drag force (parasitic + induced)."""
        parasitic_drag = self.parasitic_drag_force(velocity, position, environment)
        induced_drag = self.induced_drag_force(velocity, position, environment)
        return parasitic_drag + induced_drag

    def thrust_force(self, thrust_direction: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        """Returns thrust force vector (NED) based on throttle setting."""
        thrust_magnitude = self.propulsion.thrust(throttle)
        return thrust_magnitude * Vector.unit_vector(thrust_direction)

        
class Propulsion:
    """Models engine thrust, fuel consumption, and mass updates."""

    def __init__(
        self,
        engine_type: str,  # "turbofan", "turbojet", "piston", etc.
        num_engines: int,
        max_thrust_per_engine: float,  # Newtons
        bypass_ratio: float = None,  # For turbofans
        mass_flow_rate: float = None,  # kg/s (if known)
        specific_fuel_consumption: float = None,  # kg/N/hr (or kg/kN/hr)
        engine_efficiency: float = None,
        empty_weight: float = None,  # kg
        fuel_weight: float = None,  # kg
    ):
        self.engine_type = engine_type
        self.num_engines = num_engines
        self.max_thrust_per_engine = max_thrust_per_engine
        self.bypass_ratio = bypass_ratio
        self.mass_flow_rate = mass_flow_rate
        self.specific_fuel_consumption = specific_fuel_consumption  # kg/N/hr
        self.engine_efficiency = engine_efficiency
        self.empty_weight = empty_weight  # kg
        self.fuel_weight = fuel_weight  # kg
        self.total_mass = empty_weight + fuel_weight  # kg

    def thrust(self, throttle: float = 1.0) -> float:
        """Calculates total thrust (N) based on throttle setting (0-1)."""
        thrust_per_engine = self.max_thrust_per_engine * throttle
        return thrust_per_engine * self.num_engines

    def fuel_flow_rate(self, thrust: float) -> float:
        """Calculates fuel flow rate (kg/s) based on thrust and SFC."""
        if self.specific_fuel_consumption is None:
            raise ValueError("Specific fuel consumption (SFC) not provided.")
        # Convert SFC from kg/N/hr to kg/N/s
        sfc_per_second = self.specific_fuel_consumption / 3600
        return thrust * sfc_per_second

    def update_mass(self, dt: float, thrust: float):
        """Updates total mass (kg) based on fuel consumption over time step dt (seconds)."""
        fuel_used = self.fuel_flow_rate(thrust) * dt
        self.fuel_weight -= fuel_used
        self.total_mass = self.empty_weight + self.fuel_weight
        return self.total_mass

# Generate synthetic data for flight dynamics
n_samples = 10000
n_features = 5  # altitude, velocity, angle of attack, throttle, mass
noise = 0.1

# Generate input features (X) and target outputs (y)
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=noise,
    random_state=42,
)

# Customize the dataset to match your problem
# For example, scale the features to realistic ranges
X[:, 0] = X[:, 0] * 10000  # Altitude: 0 to 10,000 meters
X[:, 1] = X[:, 1] * 300    # Velocity: 0 to 300 m/s
X[:, 2] = X[:, 2] * 0.5    # Angle of attack: -0.5 to 0.5 radians
X[:, 3] = (X[:, 3] + 1) / 2  # Throttle: 0 to 1
X[:, 4] = X[:, 4] * 50000 + 50000  # Mass: 50,000 to 100,000 kg

# Customize the target outputs (e.g., lift, drag, thrust)
y = np.column_stack([
    y * 100000,  # Lift: scale to realistic range
    y * 50000,   # Drag: scale to realistic range
    y * 200000,  # Thrust: scale to realistic range
])

print("Input features shape:", X.shape)
print("Target outputs shape:", y.shape)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the MLP class
class FlightDynamicsMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FlightDynamicsMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize the MLP
input_size = X.shape[1]
hidden_size = 64
output_size = y.shape[1]
model = FlightDynamicsMLP(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "flight_dynamics_mlp.pth")
