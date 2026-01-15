# Vision One - Quick Reference Card

## 🎮 G-Code Commands

### Calibration Commands
```gcode
CALIBRATE_SKEW_VISION           # Auto-detect ChArUco board, calculate skew
CALIBRATE_NOZZLE_VISION         # Detect reference marker, calculate nozzle offset
CALIBRATE_FLOW_VISION EXPECTED_WIDTH=0.4  # Measure line width, suggest multiplier
VISUAL_Z_OFFSET DURATION=30     # Stream camera with crosshair overlay
```

### AI Monitoring Commands
```gcode
DETECT_SPAGHETTI PAUSE_ON_DETECT=1  # Run spaghetti detection (manual)
INSPECT_FIRST_LAYER                 # Analyze first layer quality
```

### Timelapse Commands
```gcode
START_TIMELAPSE                 # Begin capturing timelapse
STOP_TIMELAPSE                  # Stop and compile video
```

### Status Command
```gcode
VISION_STATUS                   # Show plugin status and configuration
```

---

## ⚙️ Configuration Template

```ini
[vision_one]
# Camera
camera_device: 0
camera_width: 1920
camera_height: 1080
camera_fps: 30

# Paths
base_path: /tmp/vision_one

# Features
enable_spaghetti_detection: True
enable_first_layer_inspection: True
enable_clog_detection: False
enable_timelapse: True

# Detection Parameters
spaghetti_check_interval: 10.0
spaghetti_threshold: 0.7
clog_flow_threshold: 2.0

# Timelapse
timelapse_park_x: 10.0
timelapse_park_y: 10.0
timelapse_park_z_lift: 0.5
```

---

## 🔧 Installation Commands

```bash
# Install dependencies
~/klippy-env/bin/pip install -r requirements.txt

# Copy plugin
sudo cp vision_one.py ~/klipper/klippy/extras/

# Install system packages
sudo apt install v4l-utils ffmpeg

# Restart Klipper
sudo systemctl restart klipper
```

---

## 📸 Calibration Workflow

### 1. Skew Calibration
```gcode
G28                              # Home all axes
G0 X150 Y150 Z50 F3000          # Move over ChArUco board
CALIBRATE_SKEW_VISION           # Run calibration
# Add output to [skew_correction] in printer.cfg
```

### 2. Nozzle Offset
```gcode
G0 X10 Y10 Z10 F3000            # Move over reference marker
CALIBRATE_NOZZLE_VISION         # Calculate offset
# Note the suggested X/Y positions
```

### 3. Flow Rate
```gcode
# Print single-wall test cube or line pattern
# Move camera over printed line
CALIBRATE_FLOW_VISION EXPECTED_WIDTH=0.4
# Note suggested extrusion_multiplier
```

### 4. Z-Offset
```gcode
G0 X150 Y150 Z0.2 F3000         # Move close to bed
VISUAL_Z_OFFSET DURATION=30     # Capture frames
# Review frames in /tmp/vision_one/calibration/z_offset/
# Adjust Z based on visual feedback
```

---

## 🤖 AI Usage Patterns

### Automatic Spaghetti Monitoring
```ini
# Enable in config
enable_spaghetti_detection: True
spaghetti_check_interval: 10.0

# Runs automatically during prints
# Auto-pauses if spaghetti detected
```

### First Layer Check Macro
```gcode
[gcode_macro CHECK_FIRST_LAYER_AUTO]
gcode:
    G4 P5000                    # Wait 5 seconds after layer 1
    INSPECT_FIRST_LAYER         # Run inspection
    # Add conditional logic based on result
```

---

## 🎬 Timelapse Integration

### In PRINT_START
```gcode
[gcode_macro PRINT_START]
gcode:
    # ... existing start code ...
    START_TIMELAPSE             # Begin recording
```

### In PRINT_END
```gcode
[gcode_macro PRINT_END]
gcode:
    # ... existing end code ...
    STOP_TIMELAPSE              # Compile video
```

---

## 🐛 Troubleshooting Quick Fixes

### Camera Not Detected
```bash
ls /dev/video*                  # Check devices
v4l2-ctl --list-devices        # Verify camera
sudo usermod -a -G video pi    # Add to video group
sudo reboot                     # Apply changes
```

### TFLite Errors
```bash
# Disable AI features in printer.cfg
enable_spaghetti_detection: False
enable_first_layer_inspection: False
```

### Timer Too Close
```ini
# Reduce camera load
camera_fps: 15
spaghetti_check_interval: 20.0
```

### FFMPEG Missing
```bash
sudo apt update
sudo apt install ffmpeg
pip install ffmpeg-python
```

---

## 📊 File Locations

### Plugin Files
- Plugin: `~/klipper/klippy/extras/vision_one.py`
- Config: `~/printer_data/config/printer.cfg`

### Data Directories
- Base: `/tmp/vision_one/`
- Models: `/tmp/vision_one/models/`
- Calibrations: `/tmp/vision_one/calibration/`
- Timelapses: `/tmp/vision_one/timelapse/`

### Logs
- Klipper: `~/printer_data/logs/klippy.log`
- System: `/var/log/syslog`

---

## 🔍 Debugging Commands

### Check Camera
```bash
v4l2-ctl --device=/dev/video0 --all
ffplay /dev/video0              # Test live view
```

### Monitor Logs
```bash
tail -f ~/printer_data/logs/klippy.log | grep -i vision
```

### Test Python Environment
```bash
~/klippy-env/bin/python -c "import cv2; print(cv2.__version__)"
~/klippy-env/bin/python -c "import tflite_runtime; print('TFLite OK')"
```

---

## 📈 Performance Tuning

### High Performance (RPi4 4GB+)
```ini
camera_width: 1920
camera_height: 1080
camera_fps: 30
spaghetti_check_interval: 10.0
```

### Balanced (RPi4 2GB)
```ini
camera_width: 1280
camera_height: 720
camera_fps: 20
spaghetti_check_interval: 15.0
```

### Low Performance (Limited resources)
```ini
camera_width: 640
camera_height: 480
camera_fps: 10
spaghetti_check_interval: 30.0
```

---

## 🎯 Common Use Cases

### Complete Auto-Calibration Macro
```gcode
[gcode_macro AUTO_CALIBRATE_VISION]
gcode:
    G28                         # Home
    
    # Skew
    G0 X150 Y150 Z50 F3000
    CALIBRATE_SKEW_VISION
    G4 P2000
    
    # Nozzle offset
    G0 X10 Y10 Z10 F3000
    CALIBRATE_NOZZLE_VISION
    G4 P2000
    
    # Done
    G28
    G0 X150 Y150 Z50 F3000
```

### Smart Print Start with Vision
```gcode
[gcode_macro SMART_PRINT_START]
gcode:
    # Standard startup
    G28
    G90
    M109 S{params.EXTRUDER_TEMP}
    M190 S{params.BED_TEMP}
    
    # Vision features
    START_TIMELAPSE
    # Spaghetti detection auto-starts
    
    # Begin print
```

---

## 💡 Pro Tips

1. **ChArUco Board:** Print at 100% scale, verify with ruler
2. **Lighting:** Consistent, diffuse lighting works best
3. **Camera Focus:** Manual focus, lock at working distance
4. **Park Position:** Test manually before enabling timelapse
5. **AI Models:** Start with pre-trained, fine-tune for your setup
6. **False Positives:** Adjust threshold if too many false alarms
7. **Data Collection:** Save uncertain predictions for retraining

---

## 📞 Getting Help

1. Check `VISION_STATUS` output
2. Review Klipper logs
3. Verify camera with system tools
4. Test features individually
5. Open issue with logs and configuration

---

**Print Date:** 2026-01-14
**Version:** 1.0
**Platform:** Raspberry Pi 4 + Klipper
