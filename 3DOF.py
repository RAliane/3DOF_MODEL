"""
Module for implementing a neural network (Multi-Layer Perceptron)
that backpropagates through environmental, rigid body dynamics,
trajectory, and wear-and-tear models.
"""
import math
import numpy as np
import pandas as pd
import torch
import seaborn as sns

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
