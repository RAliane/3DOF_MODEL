# 3DOF Flight Dynamics Model with Environmental and Thermodynamic Modeling

**A Python-based educational project for simulating 3DOF flight dynamics, with plans to expand to 6DOF.**

---

## Overview
This project implements a **Three Degrees of Freedom (3DOF) flight dynamics model** with environmental and thermodynamic modeling. The goal is to simulate aircraft motion using **North-East-Down (NED) coordinates** and integrate environmental effects like dynamic pressure, air density, and temperature variations with altitude.

This is a **learning project**, so expect iterative updates as I expand the model to 6DOF and refine the physics.

---

## Features
- **3DOF Flight Dynamics**: Position, velocity, and acceleration in NED coordinates.
- **Environmental Modeling**: ISA (International Standard Atmosphere) for troposphere/stratosphere effects.
- **Thermodynamic Modeling**: Dynamic pressure, air density, and temperature calculations.
- **Modular Design**: Separate `Vector` and `Environment` classes for clarity and reusability.

---

## Classes
### `Vector`
Handles 3D vectors (position, velocity, acceleration) in NED coordinates.
- Methods: `position_vector`, `velocity_vector`, `acceleration_vector`, `magnitude`.

### `Environment`
Models atmospheric conditions (density, pressure, temperature) using ISA standards.
- Methods: `dynamic_pressure`, `_temperature`, `_pressure`, `density`.

---

## Usage Example
```python
# Define position (5000 m altitude) and velocity (50 m/s north, 50 m/s east)
position = Vector.position_vector(0, 0, -5000)
velocity = Vector.velocity_vector(50, 50, 0)

# Calculate dynamic pressure
q = Environment.dynamic_pressure(position, velocity)
print(f"Dynamic pressure: {q:.2f} Pa")
