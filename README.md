   # Aerospace Flight Dynamics Model with Data-Driven MLP
   
![Python](https://img.shields.io/badge/Python-3.9-green.svg?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-green.svg?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-green.svg?style=flat-square)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Latest-yellow.svg?style=flat-square)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)

   A **3DOF/6DOF flight dynamics model** with environmental, thermodynamic, and aerodynamic modeling, enhanced with a **PyTorch MLP** for data-driven predictions. This educational project focuses on realistic flight simulation and neural network integration.
   
   ---
   
   ## Features
   - **3DOF/6DOF Flight Dynamics**: Simulates aircraft motion using NED coordinates
   - **Data-Driven MLP**: PyTorch neural network trained on synthetic flight data
   - **Realistic Data Generation**: Uses ISA atmosphere and aerodynamic models
   - **Visualization**: Plots predictions vs ground truth for lift, drag, thrust
   
   ---
   
   ## Core Components
   | Class | Purpose |
   |-------|---------|
   | `Vector` | 3D vector operations in NED coordinates |
   | `Environment` | ISA atmospheric modeling |
   | `Aerodynamics` | Lift/drag/thrust calculations |
   | `Propulsion` | Engine performance and fuel consumption |
   | `FlightDynamicsModel` | Data generation, MLP training, visualization |
   
   ---
   
   ## Quick Start
   
   ### 1. Install
   ```bash
   git clone https://github.com/RAliane/3DOF_MODEL.git
   cd 3DOF_MODEL
   pip install -r requirements.txt
   ```
   
   ### 2. Generate Data & Train
   ```python
   from 3DOF_MODEL import FlightDynamicsModel
   
   # Initialize and generate data
   fdm = FlightDynamicsModel(n_samples=5000)
   fdm.generate_data()
   
   # Train MLP
   fdm.train_mlp(hidden_layers=[128, 64, 32], dropout=0.2, lr=0.001, epochs=50)
   
   # Visualize results
   fdm.visualize_results()
   ```
   
   ### 3. Example Prediction
   ```python
   import numpy as np
   import torch
   
   # Input: [altitude, velocity, angle_of_attack, throttle]
   new_input = np.array([[8000, 250, 0.2, 0.8]])
   new_input_tensor = torch.FloatTensor(new_input)
   
   with torch.no_grad():
       lift, drag, thrust, fuel_flow = fdm.model(new_input_tensor).numpy()[0]
   
   print(f"Predictions:\nLift: {lift:.2f}N\nDrag: {drag:.2f}N\nThrust: {thrust:.2f}N\nFuel Flow: {fuel_flow:.2f}kg/s")
   ```
   
   ---
   
   ## Example Results
   ![Lift Prediction](assets/lift_plot.png)
   ![Drag Prediction](assets/drag_plot.png)
   
   *Note: Add your plots to the assets/ folder*
   
   ---
   
   ## Future Work
   - [ ] 6DOF expansion with roll/pitch/yaw
   - [ ] Advanced aerodynamics with stability derivatives
   - [ ] Time-series models for trajectory prediction
   - [ ] Real-world data validation
   
   ---
   Apache 2.0 License © Rayan Aliane
   ```
   
   
