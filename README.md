# Flight Dynamics Model with Data-Driven Aerodynamics

A **3DOF flight dynamics model** with environmental, thermodynamic, and aerodynamic modeling, now enhanced with a **data-driven MLP** trained on synthetic flight data.

---

## Features
- **Physics-Based Simulation**: Uses ISA atmosphere, NED coordinates, and aerodynamic forces (lift, drag, thrust).
- **Data-Driven MLP**: PyTorch neural network trained on synthetic flight data for real-time predictions.
- **Realistic Data Generation**: Leverages `Environment` and `Aerodynamics` classes to create physically accurate datasets.
- **Visualization**: Plots predictions vs. ground truth for lift, drag, thrust, and fuel flow.

---

## Key Classes
| Class | Purpose |
|-------|---------|
| `Vector` | Handles 3D vectors (position, velocity, acceleration) in NED coordinates. |
| `Environment` | Models atmospheric conditions (density, pressure, temperature) using ISA. |
| `Aerodynamics` | Computes lift, drag, and thrust forces. |
| `Propulsion` | Simulates engine performance, fuel consumption, and mass updates. |
| `FlightDynamicsModel` | Encapsulates data generation, MLP training, and visualization. |

---

## Data-Driven Workflow
1. **Generate Synthetic Data**:
   ```python
   fdm = FlightDynamicsModel(n_samples=5000)
   fdm.generate_data()  # Uses Environment/Aerodynamics for realism
