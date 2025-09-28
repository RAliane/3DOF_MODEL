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

# --- New: FlightDynamicsModel Class ---
class FlightDynamicsModel:
    """Encapsulates data generation, MLP training, and visualization."""

    def __init__(self, n_samples=10000, test_size=0.2, random_state=42):
        self.n_samples = n_samples
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate_data(self):
        """Generate realistic flight data using Environment and Aerodynamics."""
        X = []
        y = []
        env = Environment()
        propulsion = Propulsion(
            engine_type="turbofan",
            num_engines=2,
            max_thrust_per_engine=140000,
            specific_fuel_consumption=0.00005,
            empty_weight=50000,
            fuel_weight=20000,
        )
        aero = Aerodynamics(
            wing_area=20,
            aspect_ratio=8,
            oswald_efficiency=0.8,
            Cd0=0.02,
            Cl=0.8,
            propulsion=propulsion,
        )

        for _ in range(self.n_samples):
            # Random inputs
            altitude = np.random.uniform(0, 10000)
            velocity = np.random.uniform(50, 300)
            angle_of_attack = np.random.uniform(-0.1, 0.5)
            throttle = np.random.uniform(0.5, 1.0)

            # Create vectors
            position = Vector.position_vector(0, 0, -altitude)
            velocity_vec = Vector.velocity_vector(velocity, 0, 0)

            # Calculate outputs using your classes
            q = env.dynamic_pressure(position, velocity_vec)
            lift = aero.lift_force(velocity_vec, position, env)
            drag = aero.total_drag_force(velocity_vec, position, env)
            thrust = aero.thrust_force(Vector.position_vector(1, 0, 0), throttle)
            fuel_flow = propulsion.fuel_flow_rate(thrust[0])

            X.append([altitude, velocity, angle_of_attack, throttle])
            y.append([lift[0], drag[0], thrust[0], fuel_flow])

        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_mlp(self, hidden_layers=[128, 64, 32], dropout=0.2, lr=0.001, epochs=100):
        """Train an MLP with dropout and batch normalization."""
        input_size = self.X_train.shape[1]
        output_size = self.y_train.shape[1]

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_size = input_size
                for i, hidden_size in enumerate(hidden_layers):
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.BatchNorm1d(hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, output_size))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        self.model = MLP()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train)
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def visualize_results(self):
        """Plot predictions vs. ground truth."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_mlp() first.")

        X_test_tensor = torch.FloatTensor(self.X_test)
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).numpy()

        plt.figure(figsize=(12, 8))
        labels = ["Lift", "Drag", "Thrust", "Fuel Flow"]
        for i in range(self.y_test.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.scatter(self.y_test[:, i], y_pred[:, i], alpha=0.5)
            plt.plot([min(self.y_test[:, i]), max(self.y_test[:, i])],
                     [min(self.y_test[:, i]), max(self.y_test[:, i])], "r--")
            plt.xlabel(f"True {labels[i]}")
            plt.ylabel(f"Predicted {labels[i]}")
            plt.title(f"{labels[i]}: True vs. Predicted")
        plt.tight_layout()
        plt.show()

# --- Usage Example ---
if __name__ == "__main__":
    fdm = FlightDynamicsModel(n_samples=5000)
    fdm.generate_data()
    fdm.train_mlp(hidden_layers=[128, 64, 32], dropout=0.2, lr=0.001, epochs=50)
    fdm.visualize_results()
