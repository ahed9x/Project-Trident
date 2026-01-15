# Klipper Vision One

**Advanced Computer Vision & AI Plugin for Klipper 3D Printer Firmware**

A comprehensive Klipper plugin that brings computer vision calibration, AI-based monitoring, and timelapse capabilities to your 3D printer using a toolhead-mounted USB endoscope.

---

## 🎯 Features

### 1️⃣ Deterministic Computer Vision (OpenCV/ArUco)

#### **Automated Skew Correction**
- Detects ChArUco calibration board on the print bed
- Calculates skew_correction profile (radians) automatically
- Provides rotation_distance scale factors for X/Y axes
- **Command:** `CALIBRATE_SKEW_VISION`

#### **Visual Nozzle Alignment**
- Detects circular reference markers or bed holes
- Calculates absolute X/Y nozzle offsets with pixel-perfect precision
- **Command:** `CALIBRATE_NOZZLE_VISION`

#### **Flow Rate Calibration**
- Analyzes printed test patterns using edge detection
- Measures line width in microns
- Suggests optimal extrusion_multiplier values
- **Command:** `CALIBRATE_FLOW_VISION EXPECTED_WIDTH=0.4`

#### **Visual Z-Offset**
- Streams camera feed with digital crosshair overlay
- Enables manual precision leveling with visual feedback
- **Command:** `VISUAL_Z_OFFSET [DURATION=30]`

### 2️⃣ AI & Machine Learning (TensorFlow Lite)

#### **Spaghetti Detection**
- Background thread runs lightweight TFLite classification every 10 seconds
- Automatically pauses print when spaghetti is detected
- Configurable confidence threshold
- **Command:** `DETECT_SPAGHETTI [PAUSE_ON_DETECT=1]`

#### **First Layer Inspection**
- Binary classifier analyzes first layer texture (Good/Bad)
- Provides quality score and recommendations
- **Command:** `INSPECT_FIRST_LAYER`

#### **Clog Detection** *(Experimental)*
- Monitors optical flow of filament exiting nozzle
- Triggers PAUSE if motion stops while extruder is active
- Real-time monitoring during prints

### 3️⃣ Media Features

#### **Stabilized Timelapse**
- Hooks into layer change events
- Parks toolhead to consistent position
- Captures frame and appends to video file
- Uses FFMPEG for video compilation
- **Commands:** `START_TIMELAPSE`, `STOP_TIMELAPSE`

---

## 📁 Project Structure

```
vision_one/
├── vision_one.py              # Main plugin code (place in ~/klipper/klippy/extras/)
├── requirements.txt           # Python dependencies
├── printer.cfg.example        # Configuration example
├── README.md                  # This file
├── models/                    # AI/ML models directory
│   ├── spaghetti_detector.tflite
│   └── first_layer_classifier.tflite
├── calibration/              # Calibration data storage
└── timelapse/                # Timelapse frames and videos
```

---

## 🚀 Installation

### Prerequisites
- **Hardware:**
  - Raspberry Pi 4 (or similar) running Klipper/Moonraker
  - USB Endoscope (5.5mm or similar) mounted on toolhead
  - Voron Trident or compatible 3D printer

- **Software:**
  - Klipper firmware installed
  - Python 3.7+ (included with Klipper environment)

### Step 1: Copy Plugin to Klipper

```bash
# Navigate to your Klipper extras directory
cd ~/klipper/klippy/extras/

# Copy the vision_one.py file
sudo cp /path/to/vision_one.py ./

# Set appropriate permissions
sudo chown pi:pi vision_one.py
sudo chmod 644 vision_one.py
```

### Step 2: Install Python Dependencies

```bash
# Activate Klipper's Python environment
source ~/klippy-env/bin/activate

# Install dependencies
pip install -r /path/to/requirements.txt

# Deactivate environment
deactivate
```

### Step 3: Configure printer.cfg

Add the Vision One configuration to your `printer.cfg`:

```ini
[vision_one]
camera_device: 0
camera_width: 1920
camera_height: 1080
camera_fps: 30
base_path: /tmp/vision_one

enable_spaghetti_detection: True
enable_first_layer_inspection: True
enable_clog_detection: False
enable_timelapse: True

spaghetti_check_interval: 10.0
spaghetti_threshold: 0.7

timelapse_park_x: 10.0
timelapse_park_y: 10.0
timelapse_park_z_lift: 0.5
```

See `printer.cfg.example` for complete configuration options.

### Step 4: (Optional) Add AI Models

Place your TensorFlow Lite models in the models directory:

```bash
mkdir -p /tmp/vision_one/models/
cp spaghetti_detector.tflite /tmp/vision_one/models/
cp first_layer_classifier.tflite /tmp/vision_one/models/
```

**Note:** AI models are optional. Vision calibration features work without them.

### Step 5: Restart Klipper

```bash
sudo systemctl restart klipper
```

### Step 6: Verify Installation

In your Klipper console (Mainsail/Fluidd/OctoPrint):

```gcode
VISION_STATUS
```

You should see output confirming camera detection and enabled features.

---

## 📖 Usage Guide

### Basic Calibration Workflow

#### 1. Skew Calibration
```gcode
# Print a ChArUco calibration board (7x5 squares, 40mm square, 30mm marker)
# Place on print bed
# Move toolhead over board
G0 X150 Y150 Z50 F3000

# Run calibration
CALIBRATE_SKEW_VISION

# Results will show skew angle and scale factors
# Add suggested values to printer.cfg [skew_correction] section
```

#### 2. Nozzle Offset Calibration
```gcode
# Move over reference marker (circular hole or marker)
G0 X10 Y10 Z10 F3000

# Run calibration
CALIBRATE_NOZZLE_VISION

# Results show calculated X/Y offsets
```

#### 3. Flow Rate Calibration
```gcode
# Print a single-line test pattern
# Move camera over the printed line

# Run calibration (specify expected line width)
CALIBRATE_FLOW_VISION EXPECTED_WIDTH=0.4

# Results suggest new extrusion multiplier
```

#### 4. Visual Z-Offset
```gcode
# Move nozzle close to bed
G0 X150 Y150 Z0.3 F3000

# Start visual stream (saves frames to /tmp/vision_one/calibration/z_offset/)
VISUAL_Z_OFFSET DURATION=30

# Adjust Z while viewing crosshair overlay on frames
# Review captured frames to verify Z-offset
```

### AI Monitoring

#### Manual Spaghetti Detection
```gcode
DETECT_SPAGHETTI PAUSE_ON_DETECT=1
```

#### First Layer Inspection
```gcode
# Run after first layer completes
INSPECT_FIRST_LAYER
```

#### Automatic Monitoring
When `enable_spaghetti_detection: True`, the plugin automatically checks every 10 seconds during prints. No manual commands needed!

### Timelapse Recording

#### Manual Control
```gcode
# Start recording
START_TIMELAPSE

# ... print happens ...

# Stop and compile video
STOP_TIMELAPSE
```

#### Automatic with Print Macros
Add to your `PRINT_START` macro:
```gcode
[gcode_macro PRINT_START]
gcode:
    # ... existing start code ...
    START_TIMELAPSE
```

Add to your `PRINT_END` macro:
```gcode
[gcode_macro PRINT_END]
gcode:
    # ... existing end code ...
    STOP_TIMELAPSE
```

---

## ⚙️ Configuration Reference

### Camera Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_device` | 0 | USB camera device ID (/dev/videoN) |
| `camera_width` | 1920 | Camera resolution width |
| `camera_height` | 1080 | Camera resolution height |
| `camera_fps` | 30 | Camera frames per second |

### Feature Flags
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_spaghetti_detection` | False | Enable AI spaghetti detection |
| `enable_first_layer_inspection` | False | Enable first layer AI analysis |
| `enable_clog_detection` | False | Enable optical flow clog detection |
| `enable_timelapse` | False | Enable timelapse recording |

### AI Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `spaghetti_check_interval` | 10.0 | Seconds between spaghetti checks |
| `spaghetti_threshold` | 0.7 | Confidence threshold (0.0-1.0) |
| `clog_flow_threshold` | 2.0 | Minimum optical flow magnitude |

### Timelapse Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `timelapse_park_x` | None | X position to park for frame capture |
| `timelapse_park_y` | None | Y position to park for frame capture |
| `timelapse_park_z_lift` | 0.5 | Z-axis lift during park (mm) |

---

## 🧠 AI Model Training (Advanced)

The plugin supports custom TensorFlow Lite models. Here's how to train your own:

### Spaghetti Detection Model

1. **Collect Dataset:**
   - Capture images of normal prints (Class 0)
   - Capture images of failed prints with spaghetti (Class 1)
   - Minimum 1000 images per class recommended

2. **Train Model:**
   ```python
   # Use MobileNetV2 or similar lightweight architecture
   # Input shape: (224, 224, 3)
   # Output: Binary classification (0=Normal, 1=Spaghetti)
   ```

3. **Convert to TFLite:**
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   
   with open('spaghetti_detector.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

4. **Deploy:**
   ```bash
   cp spaghetti_detector.tflite /tmp/vision_one/models/
   ```

### First Layer Inspection Model

Similar process, but classify first layer quality:
- Class 0: Poor adhesion/quality
- Class 1: Good adhesion/quality

---

## 🔧 Troubleshooting

### Camera Not Detected
```bash
# Check available cameras
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Verify permissions
sudo usermod -a -G video pi
```

### TFLite Import Error
```bash
# Install TFLite runtime
pip install tflite-runtime==2.14.0

# Or use full TensorFlow (larger, slower)
pip install tensorflow==2.14.0
```

### "Timer Too Close" Errors
The plugin uses background threads to avoid blocking Klipper. If you still see timer errors:
1. Reduce `camera_fps` (e.g., 15 instead of 30)
2. Increase `spaghetti_check_interval`
3. Disable features you're not using

### FFMPEG Not Found
```bash
# Install FFMPEG
sudo apt update
sudo apt install ffmpeg

# Install Python bindings
pip install ffmpeg-python
```

---

## 🛠️ Architecture Details

### Asynchronous Design
- **Camera Thread:** Continuously captures frames without blocking Klipper
- **AI Monitor Thread:** Runs inference at configurable intervals
- **Frame Queue:** Decouples capture from processing (max 5 frames buffered)
- **Reactor Callbacks:** All Klipper commands use reactor for safe execution

### Thread Safety
- All camera access protected by `camera_lock`
- G-code commands use reactor callbacks for thread-safe execution
- Frame queue prevents memory overflow

### Performance Optimization
- OpenCV compiled without GUI (headless) for smaller footprint
- TFLite runtime instead of full TensorFlow
- Lazy loading of AI models (only when features enabled)
- Frame downsampling before inference

---

## 📝 G-Code Command Reference

| Command | Parameters | Description |
|---------|------------|-------------|
| `CALIBRATE_SKEW_VISION` | None | Detect ChArUco board, calculate skew |
| `CALIBRATE_NOZZLE_VISION` | None | Detect marker, calculate nozzle offset |
| `CALIBRATE_FLOW_VISION` | `EXPECTED_WIDTH=0.4` | Analyze line width, suggest multiplier |
| `VISUAL_Z_OFFSET` | `DURATION=30` | Stream camera with crosshair overlay |
| `DETECT_SPAGHETTI` | `PAUSE_ON_DETECT=1` | Run spaghetti detection |
| `INSPECT_FIRST_LAYER` | None | Analyze first layer quality |
| `START_TIMELAPSE` | None | Begin timelapse recording |
| `STOP_TIMELAPSE` | None | Stop and compile timelapse video |
| `VISION_STATUS` | None | Display status and configuration |

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional calibration routines (bed mesh visualization, etc.)
- More AI models (layer adhesion, warping detection)
- Real-time streaming to web interface
- Integration with Moonraker/Mainsail

---

## 📄 License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This is experimental software. Always monitor your printer during operation. The AI detection features are not 100% accurate and should not be relied upon as the sole safety mechanism.

---

## 🙏 Acknowledgments

- Klipper Firmware Team
- OpenCV & ArUco Community
- TensorFlow Lite Team
- Voron Design Community

---

## 📞 Support

For issues, questions, or feature requests, please open an issue on the project repository.

**Happy Printing! 🎉**
