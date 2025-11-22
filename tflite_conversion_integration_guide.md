# TensorFlow Lite Conversion & LQRI-PID Integration Guide

## Overview

This guide integrates your 3DoF TensorFlow model into the 6DoF flight dynamics system via a TensorFlow Lite converter, creating a hybrid control system:

```
TensorFlow Neural Network (3DoF)
    ↓ (Convert to TFLite)
    ↓
Lightweight TFLite Model
    ↓ (Feedforward Prediction)
    ↓
LQRI-PID Controller (On ESP32)
    ↓
6DoF Flight Dynamics
    ↓
UDP Telemetry → ESP8266 → OLED Display
```

---

## Step 1: Prepare Your TensorFlow Model

### Requirements
```bash
pip install tensorflow>=2.10.0
pip install numpy pandas scikit-learn
```

### Your 3DoF Model Structure

Your model should take as input:
- `altitude` (meters)
- `airspeed` (m/s)  
- `angle_of_attack` (radians)
- `throttle_setting` (0.0-1.0)

Output should be:
- `throttle_adjustment` (-0.1 to 0.1)
- `elevator_deflection` (-0.1 to 0.1)
- `aileron_deflection` (-0.1 to 0.1)

### Example: Building Your 3DoF Model

```python
import tensorflow as tf
import numpy as np

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3)  # Output: [throttle, elevator, aileron]
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train with your data
# X_train: [altitude, airspeed, aoa, throttle]
# y_train: [throttle_adj, elevator_adj, aileron_adj]
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save model
model.save('flight_control_3dof.h5')
```

---

## Step 2: Convert to TensorFlow Lite

### Step 2A: Basic Conversion

```python
from lqri_pid_tflite import TFLiteModelConverter

# Convert with dynamic quantization (smallest size)
TFLiteModelConverter.convert_model_to_tflite(
    keras_model_path='flight_control_3dof.h5',
    output_path='flight_control_3dof.tflite',
    quantization='dynamic'
)
```

### Step 2B: Advanced Quantization Options

#### Option 1: Full Integer Quantization (Smallest)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Requires representative dataset
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
```

#### Option 2: Float16 Quantization (Balance)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### Step 2C: Verify Converted Model

```python
from lqri_pid_tflite import TFLiteInterpreter

# Load and test
interpreter = TFLiteInterpreter('flight_control_3dof.tflite')

# Get model info
info = interpreter.get_model_info()
print(f"Input shape: {info['input_shape']}")
print(f"Output shape: {info['output_shape']}")

# Test inference
test_input = np.array([[
    5000.0,    # altitude (m)
    150.0,     # airspeed (m/s)
    0.05,      # angle of attack (rad)
    0.6        # throttle
]], dtype=np.float32)

output = interpreter.predict(test_input)
print(f"Output: {output}")
```

---

## Step 3: Deploy to ESP32

### Step 3A: Prepare Model for Embedded Use

Option 1: Store as Base64 String (Easiest)
```python
import base64

with open('flight_control_3dof.tflite', 'rb') as f:
    model_bytes = f.read()
    
model_base64 = base64.b64encode(model_bytes).decode('utf-8')

# Save to file
with open('model_embedded.h', 'w') as f:
    f.write(f'const char MODEL_DATA[] = "{model_base64}";\n')
    f.write(f'const size_t MODEL_SIZE = {len(model_bytes)};\n')
```

Option 2: Store on SPIFFS/LittleFS (Recommended)
```bash
# Create data directory
mkdir data
cp flight_control_3dof.tflite data/

# Upload using Arduino IDE:
# Tools → ESP32 Sketch Data Upload
```

### Step 3B: Arduino Library Installation

In Arduino IDE:
- Sketch → Include Library → Manage Libraries
- Search and install:
  - `TensorFlow Lite Micro` by TensorFlow
  - `Arduino_TensorFlowLite`

### Step 3C: Load Model in ESP32 Code

```cpp
// Using SPIFFS
#include <SPIFFS.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

void loadModel() {
    SPIFFS.begin(true);
    File modelFile = SPIFFS.open("/flight_control_3dof.tflite", "r");
    
    if (!modelFile) {
        Serial.println("Failed to load model");
        return;
    }
    
    size_t modelSize = modelFile.size();
    uint8_t* modelBuffer = new uint8_t[modelSize];
    modelFile.readBytes((char*)modelBuffer, modelSize);
    modelFile.close();
    
    // Initialize TFLite interpreter
    // (See full implementation in ESP32 sketch)
}
```

---

## Step 4: LQRI-PID Controller Integration

### Architecture

```
NN Feedforward (30%)     Classical PID Feedback (70%)
        ↓                            ↓
        └─────── Combined ───────────┘
                    ↓
            Control Outputs
         (throttle, elevator, aileron, rudder)
                    ↓
            6DoF Flight Dynamics
```

### Control Law

```
throttle = 0.3 * NN_throttle + 0.7 * (PID_altitude + PID_airspeed)
elevator = 0.3 * NN_elevator + 0.7 * PID_pitch
aileron = 0.3 * NN_aileron + 0.7 * PID_roll
rudder = PID_yaw
```

### Tuning LQRI-PID Gains

```python
# PID gain structure
pid_gains = {
    'altitude': {'Kp': 0.01, 'Ki': 0.001, 'Kd': 0.002},
    'airspeed': {'Kp': 0.05, 'Ki': 0.002, 'Kd': 0.01},
    'pitch': {'Kp': 0.5, 'Ki': 0.05, 'Kd': 0.1},
    'roll': {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.08},
    'yaw': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05}
}

# Guidelines for tuning:
# Kp: Proportional gain - increase for faster response
# Ki: Integral gain - removes steady-state error, avoid overshoot
# Kd: Derivative gain - dampens oscillations
```

#### Ziegler-Nichols Method (Auto-tuning)

```python
# 1. Set Ki=0, Kd=0
# 2. Increase Kp until system oscillates
# 3. Note the gain (Ku) and period (Tu)
# 4. Calculate:
Kp = 0.6 * Ku
Ki = 1.2 * Ku / Tu
Kd = 0.075 * Ku * Tu

# 5. Fine-tune by ±20% if needed
```

### Sending Gain Updates via UDP

```python
# Node-RED function to update gains
msg.payload = JSON.stringify({
    "altitude_sp": 2000,    // Target altitude (m)
    "airspeed_sp": 120,     // Target airspeed (m/s)
    "Kp": 0.015,            // Update proportional gain
    "Ki": 0.0015,
    "Kd": 0.003
});

return msg;
```

```cpp
// ESP32 receives and applies updates
if (command.hasOwnProperty("Kp")) {
    float kp = (float)command["Kp"];
    float ki = (float)command["Ki"];
    float kd = (float)command["Kd"];
    autopilot.updateGains("altitude", kp, ki, kd);
}
```

---

## Step 5: Testing & Validation

### Phase 1: Model Verification

```python
# Ensure inference works
test_cases = [
    [1000, 100, 0.05, 0.5],  # Cruise
    [5000, 150, 0.1, 0.7],   # Climb
    [500, 80, -0.05, 0.3],   # Descent
]

for inputs in test_cases:
    output = interpreter.predict(np.array([inputs], dtype=np.float32))
    print(f"Input: {inputs} → Output: {output}")
```

### Phase 2: LQRI-PID Simulation

```python
# Simulate controller response
controller = LQRIPID('flight_control_3dof.tflite', dt=0.01)
controller.set_setpoints(altitude=2000, airspeed=120)

# Simulate for 100 seconds
for t in range(10000):
    state = ControlState(
        altitude=0 + t*0.2,  # Ascending
        airspeed=50 + t*0.007,
        pitch=np.sin(t*0.001)*0.1,
        roll=0,
        yaw=0,
        pitch_rate=0.01,
        roll_rate=0,
        yaw_rate=0
    )
    
    control = controller.compute_control(state)
    print(f"T={t*0.01:.1f}s: Alt={state.altitude:.0f}m, "
          f"Throttle={control.throttle:.2f}")
```

### Phase 3: Hardware Integration

1. **Upload to ESP32:**
   - Compile and upload LQRI-PID ESP32 sketch
   - Verify Serial Monitor shows control outputs

2. **Connect ESP8266:**
   - Upload telemetry server code
   - Monitor OLED for data reception

3. **Start Node-RED:**
   - Receive telemetry packets
   - View dashboard at localhost:1880/ui
   - Send gain update commands

### Phase 4: In-Flight Testing Sequence

```
1. Hover/Stabilize (t=0-5s)
   - All outputs zero
   - Verify no drift

2. Altitude Hold (t=5-30s)
   - Set altitude setpoint
   - Monitor controller response
   - Check overshoot/oscillation

3. Airspeed Hold (t=30-60s)
   - Set airspeed target
   - Verify throttle adjustments

4. Combined Maneuver (t=60+s)
   - Ramp altitude
   - Observe coupled effects
   - Verify all axes stable
```

---

## Performance Metrics

### Key Measurements

```
Steady-State Error: |setpoint - actual| at t=∞
Rise Time: Time to reach 90% of setpoint
Settling Time: Time to stay within ±5% of setpoint
Overshoot: Peak above setpoint / setpoint * 100%
Oscillation: Count of oscillations during settling
```

### Optimization Targets

- **Altitude**: Rise time < 5s, overshoot < 10%
- **Airspeed**: Rise time < 3s, overshoot < 5%
- **Stability**: No oscillations after settling

---

## Troubleshooting

### TFLite Conversion Issues

| Problem | Solution |
|---------|----------|
| Model too large | Use full_integer quantization |
| Inference slow | Reduce model size, use float16 |
| Accuracy drop | Verify quantization representative dataset |
| Input shape mismatch | Check model input spec in `get_model_info()` |

### LQRI-PID Issues

| Problem | Solution |
|---------|----------|
| Oscillations | Decrease Kp, increase Kd |
| Slow response | Increase Kp, increase Ki |
| Steady-state error | Increase Ki |
| Noise in output | Decrease Kd, add filtering |

### Integration Issues

| Problem | Solution |
|---------|----------|
| ESP32 won't compile | Check TFLite library version |
| Model won't load | Verify SPIFFS upload succeeded |
| NN gives wrong outputs | Check input normalization (0-1 range) |
| No UDP packets | Verify network config, check firewall |

---

## Advanced: On-Device Model Training (Optional)

For adaptive control, you can retrain the model on ESP32:

```cpp
// Lightweight TensorFlow Lite training API
// (Available in TFLite Micro with training support)

// Store trajectory history
struct TrajectoryPoint {
    float altitude, airspeed, control_out;
} trajectory[1000];

// Periodically retrain with recent data
void retrainModel() {
    // Gather recent trajectory data
    // Use TFLite Micro training to update weights
    // Only practical for small models
}
```

---

## Performance Summary

| Component | Resource Usage | Notes |
|-----------|---|---|
| TFLite Model | 2-5 MB | Quantized 3DoF model |
| LQRI-PID Controller | ~50 KB | On-board implementation |
| Control Loop | 10 ms | 100 Hz control frequency |
| Telemetry Rate | 100 ms | 10 Hz telemetry |
| ESP32 Memory | ~30% used | Remaining for future expansion |

---

## Next Steps

1. Train and convert your 3DoF model to TFLite
2. Deploy to ESP32 and verify inference
3. Tune PID gains using Ziegler-Nichols
4. Test in simulation with Node-RED dashboard
5. Validate with real sensor data
6. Deploy to hardware for flight testing

---

## References & Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [PID Tuning Methods](https://en.wikipedia.org/wiki/PID_controller#Manual_tuning)
- [Linear Quadratic Regulator](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator)
- [ESP32 TFLite Examples](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples)