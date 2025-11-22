"""
LQRI-PID Controller with TensorFlow Lite Integration for 6DoF Flight Dynamics
Combines reinforcement learning predictions with classical control

Copyright 2025 Rayan Aliane
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import tensorflow as tf
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from collections import deque


# ============================================================================
# TFLITE MODEL CONVERTER & LOADER
# ============================================================================

class TFLiteModelConverter:
    """Converts trained Keras models to TFLite for embedded deployment."""

    @staticmethod
    def convert_model_to_tflite(keras_model_path: str, output_path: str,
                                quantization: str = 'dynamic') -> str:
        """
        Convert Keras model to TFLite format with optional quantization.
        
        Args:
            keras_model_path: Path to saved Keras model
            output_path: Output path for .tflite file
            quantization: 'dynamic', 'full_integer', or 'float16'
        
        Returns:
            Path to converted .tflite model
        """
        # Load Keras model
        model = tf.keras.models.load_model(keras_model_path)
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations
        if quantization == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == 'full_integer':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted and saved to {output_path}")
        return output_path


class TFLiteInterpreter:
    """Lightweight TFLite model inference engine for embedded systems."""

    def __init__(self, model_path: str):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to .tflite model file
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input array matching model input shape
        
        Returns:
            Output predictions from model
        """
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], 
                                     input_data.astype(np.float32))
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def get_model_info(self) -> Dict:
        """Return input/output shapes and types."""
        return {
            'input_shape': self.input_details[0]['shape'],
            'input_type': self.input_details[0]['dtype'],
            'output_shape': self.output_details[0]['shape'],
            'output_type': self.output_details[0]['dtype']
        }


# ============================================================================
# LQRI-PID CONTROLLER IMPLEMENTATION
# ============================================================================

@dataclass
class ControlState:
    """State representation for flight control."""
    altitude: float           # meters
    airspeed: float          # m/s
    pitch: float             # radians
    roll: float              # radians
    yaw: float               # radians
    pitch_rate: float        # rad/s
    roll_rate: float         # rad/s
    yaw_rate: float          # rad/s
    altitude_error_integral: float = 0.0
    airspeed_error_integral: float = 0.0


@dataclass
class ControlOutput:
    """Control surface deflections."""
    throttle: float          # 0.0-1.0
    elevator_pitch: float    # -1.0 to 1.0 (nose up/down)
    aileron_roll: float      # -1.0 to 1.0 (left/right)
    rudder_yaw: float        # -1.0 to 1.0 (left/right)


class LQRIPID:
    """
    Linear Quadratic Regulator with Integral action combined with PID feedback.
    Uses neural network predictions as feedforward + classical PID feedback.
    """

    def __init__(self, tflite_model_path: str, dt: float = 0.01):
        """
        Initialize LQRI-PID controller.
        
        Args:
            tflite_model_path: Path to TFLite model for feedforward prediction
            dt: Control timestep (seconds)
        """
        self.dt = dt
        self.nn_model = TFLiteInterpreter(tflite_model_path)
        
        # Reference states (setpoints)
        self.altitude_setpoint = 1000.0  # meters
        self.airspeed_setpoint = 100.0   # m/s
        self.pitch_setpoint = 0.0        # radians
        
        # PID gains
        self.pid_gains = {
            'altitude': {'Kp': 0.01, 'Ki': 0.001, 'Kd': 0.002},
            'airspeed': {'Kp': 0.05, 'Ki': 0.002, 'Kd': 0.01},
            'pitch': {'Kp': 0.5, 'Ki': 0.05, 'Kd': 0.1},
            'roll': {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.08},
            'yaw': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05}
        }
        
        # LQR state-space matrices (simplified)
        self.lqr_gain = np.array([
            [0.1, 0.05, 0.02],  # Throttle gains
            [0.2, 0.1, 0.05],   # Elevator gains
            [0.15, 0.08, 0.03]  # Aileron gains
        ])
        
        # Error history for derivative term
        self.error_history = {
            'altitude': deque(maxlen=3),
            'airspeed': deque(maxlen=3),
            'pitch': deque(maxlen=3),
            'roll': deque(maxlen=3),
            'yaw': deque(maxlen=3)
        }
        
        # Output limits
        self.output_limits = {
            'throttle': (0.0, 1.0),
            'elevator': (-1.0, 1.0),
            'aileron': (-1.0, 1.0),
            'rudder': (-1.0, 1.0)
        }

    def set_setpoints(self, altitude: float, airspeed: float, pitch: float = 0.0):
        """Set control objectives."""
        self.altitude_setpoint = altitude
        self.airspeed_setpoint = airspeed
        self.pitch_setpoint = pitch

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

    def _compute_pid_term(self, error: float, error_history: deque, gains: Dict) -> float:
        """Compute PID term: Kp*e + Ki*integral + Kd*derivative."""
        # Proportional
        p_term = gains['Kp'] * error
        
        # Integral (accumulated error)
        i_term = gains['Ki'] * sum(error_history) * self.dt
        
        # Derivative (rate of change)
        if len(error_history) >= 2:
            d_term = gains['Kd'] * (error - error_history[-1]) / self.dt
        else:
            d_term = 0.0
        
        return p_term + i_term + d_term

    def _get_nn_feedforward(self, state: ControlState) -> np.ndarray:
        """
        Get feedforward correction from neural network.
        NN predicts optimal control adjustments based on flight state.
        """
        # Prepare input: [altitude, airspeed, pitch, roll, yaw]
        nn_input = np.array([[
            state.altitude / 10000.0,      # Normalize to ~0-1 range
            state.airspeed / 300.0,
            state.pitch,
            state.roll,
            state.yaw
        ]], dtype=np.float32)
        
        # Get NN prediction: [throttle_adjustment, elevator_adj, aileron_adj]
        nn_output = self.nn_model.predict(nn_input)
        return nn_output[0]  # Shape: (3,)

    def compute_control(self, state: ControlState) -> ControlOutput:
        """
        Main control law: Combine NN feedforward with PID feedback.
        
        Throttle = NN_throttle + PID_altitude + PID_airspeed
        Elevator = NN_elevator + PID_pitch
        Aileron = NN_aileron + PID_roll
        Rudder = PID_yaw
        """
        
        # Compute errors
        altitude_error = self.altitude_setpoint - state.altitude
        airspeed_error = self.airspeed_setpoint - state.airspeed
        pitch_error = self.pitch_setpoint - state.pitch
        roll_error = 0.0 - state.roll  # Target roll = 0
        yaw_error = 0.0 - state.yaw    # Target yaw = 0
        
        # Store errors for derivative term
        self.error_history['altitude'].append(altitude_error)
        self.error_history['airspeed'].append(airspeed_error)
        self.error_history['pitch'].append(pitch_error)
        self.error_history['roll'].append(roll_error)
        self.error_history['yaw'].append(yaw_error)
        
        # Get NN feedforward predictions
        try:
            nn_ff = self._get_nn_feedforward(state)
        except Exception as e:
            print(f"NN inference failed: {e}")
            nn_ff = np.array([0.0, 0.0, 0.0])
        
        # Compute PID feedback terms
        alt_pid = self._compute_pid_term(
            altitude_error, self.error_history['altitude'], 
            self.pid_gains['altitude']
        )
        
        airspeed_pid = self._compute_pid_term(
            airspeed_error, self.error_history['airspeed'],
            self.pid_gains['airspeed']
        )
        
        pitch_pid = self._compute_pid_term(
            pitch_error, self.error_history['pitch'],
            self.pid_gains['pitch']
        )
        
        roll_pid = self._compute_pid_term(
            roll_error, self.error_history['roll'],
            self.pid_gains['roll']
        )
        
        yaw_pid = self._compute_pid_term(
            yaw_error, self.error_history['yaw'],
            self.pid_gains['yaw']
        )
        
        # Combine feedforward (NN) + feedback (PID)
        throttle = nn_ff[0] * 0.3 + (alt_pid + airspeed_pid) * 0.7
        elevator = nn_ff[1] * 0.3 + pitch_pid * 0.7
        aileron = nn_ff[2] * 0.3 + roll_pid * 0.7
        rudder = yaw_pid
        
        # Apply output limits
        throttle = self._clamp(throttle, *self.output_limits['throttle'])
        elevator = self._clamp(elevator, *self.output_limits['elevator'])
        aileron = self._clamp(aileron, *self.output_limits['aileron'])
        rudder = self._clamp(rudder, *self.output_limits['rudder'])
        
        return ControlOutput(
            throttle=throttle,
            elevator_pitch=elevator,
            aileron_roll=aileron,
            rudder_yaw=rudder
        )

    def update_pid_gains(self, control_type: str, Kp: float, Ki: float, Kd: float):
        """Tune PID gains online."""
        if control_type in self.pid_gains:
            self.pid_gains[control_type] = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            print(f"Updated {control_type} gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")


# ============================================================================
# INTEGRATION WITH 6DOF FLIGHT DYNAMICS
# ============================================================================

class FlightDynamicsController:
    """Integrates LQRI-PID controller with 6DoF physics engine."""

    def __init__(self, tflite_model_path: str, dt: float = 0.01):
        """Initialize flight dynamics with controller."""
        self.controller = LQRIPID(tflite_model_path, dt)
        self.dt = dt
        self.state = ControlState(
            altitude=0.0,
            airspeed=0.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            pitch_rate=0.0,
            roll_rate=0.0,
            yaw_rate=0.0
        )
        self.control_history = []

    def update(self, aircraft_state: Dict) -> Dict:
        """
        Update control based on current aircraft state.
        
        Args:
            aircraft_state: Dict with keys:
                - 'altitude', 'airspeed', 'pitch', 'roll', 'yaw'
                - 'pitch_rate', 'roll_rate', 'yaw_rate'
        
        Returns:
            Control outputs dict
        """
        # Update internal state
        self.state.altitude = aircraft_state.get('altitude', 0.0)
        self.state.airspeed = aircraft_state.get('airspeed', 0.0)
        self.state.pitch = aircraft_state.get('pitch', 0.0)
        self.state.roll = aircraft_state.get('roll', 0.0)
        self.state.yaw = aircraft_state.get('yaw', 0.0)
        self.state.pitch_rate = aircraft_state.get('pitch_rate', 0.0)
        self.state.roll_rate = aircraft_state.get('roll_rate', 0.0)
        self.state.yaw_rate = aircraft_state.get('yaw_rate', 0.0)
        
        # Compute control
        control = self.controller.compute_control(self.state)
        
        # Log
        self.control_history.append({
            'timestamp': len(self.control_history) * self.dt,
            'altitude': self.state.altitude,
            'airspeed': self.state.airspeed,
            'throttle': control.throttle,
            'elevator': control.elevator_pitch,
            'aileron': control.aileron_roll,
            'rudder': control.rudder_yaw
        })
        
        return {
            'throttle': control.throttle,
            'elevator': control.elevator_pitch,
            'aileron': control.aileron_roll,
            'rudder': control.rudder_yaw
        }

    def set_target(self, altitude: float, airspeed: float):
        """Set autopilot target."""
        self.controller.set_setpoints(altitude, airspeed)

    def export_telemetry(self, filename: str = 'control_telemetry.json'):
        """Export control history to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.control_history, f, indent=2)


# ============================================================================
# USAGE EXAMPLE FOR EMBEDDED SYSTEMS
# ============================================================================

if __name__ == "__main__":
    # Example: Initialize and use LQRI-PID controller
    
    # Step 1: Convert Keras model to TFLite
    # TFLiteModelConverter.convert_model_to_tflite(
    #     'path/to/keras/model',
    #     'flight_model.tflite',
    #     quantization='dynamic'
    # )
    
    # Step 2: Initialize flight dynamics controller
    # fdc = FlightDynamicsController('flight_model.tflite', dt=0.01)
    
    # Step 3: Set autopilot target
    # fdc.set_target(altitude=5000.0, airspeed=150.0)
    
    # Step 4: Simulate control loop
    # for i in range(1000):
    #     # Get current aircraft state (from sensors)
    #     aircraft_state = {
    #         'altitude': 0.0 + i*0.5,  # Ascending
    #         'airspeed': 50.0 + i*0.05,
    #         'pitch': 0.1,
    #         'roll': 0.0,
    #         'yaw': 0.0,
    #         'pitch_rate': 0.01,
    #         'roll_rate': 0.0,
    #         'yaw_rate': 0.0
    #     }
    #     
    #     # Update control
    #     control = fdc.update(aircraft_state)
    #     print(f"Throttle: {control['throttle']:.3f}, "
    #           f"Elevator: {control['elevator']:.3f}")
    
    print("LQRI-PID Controller Module Ready for Integration")
    print("Supports TFLite models from Keras/TensorFlow")
    print("Compatible with ESP32 6DoF Flight Dynamics")