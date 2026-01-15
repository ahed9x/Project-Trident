#!/bin/bash
# Installation script for Klipper Vision One
# Run this script on your Raspberry Pi

set -e

echo "========================================="
echo "Klipper Vision One Installation Script"
echo "========================================="
echo ""

# Check if running as pi user
if [ "$USER" != "pi" ]; then
    echo "Warning: This script should be run as the 'pi' user"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Paths
KLIPPER_EXTRAS="$HOME/klipper/klippy/extras"
VISION_ONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KLIPPY_ENV="$HOME/klippy-env"

echo "Step 1: Checking Klipper installation..."
if [ ! -d "$HOME/klipper" ]; then
    echo "Error: Klipper not found at $HOME/klipper"
    exit 1
fi
echo "✓ Klipper found"

echo ""
echo "Step 2: Copying vision_one.py to Klipper extras..."
if [ ! -f "$VISION_ONE_DIR/vision_one.py" ]; then
    echo "Error: vision_one.py not found in current directory"
    exit 1
fi

sudo cp "$VISION_ONE_DIR/vision_one.py" "$KLIPPER_EXTRAS/"
sudo chown pi:pi "$KLIPPER_EXTRAS/vision_one.py"
sudo chmod 644 "$KLIPPER_EXTRAS/vision_one.py"
echo "✓ Plugin copied"

echo ""
echo "Step 3: Installing Python dependencies..."
if [ ! -d "$KLIPPY_ENV" ]; then
    echo "Error: Klipper Python environment not found at $KLIPPY_ENV"
    exit 1
fi

source "$KLIPPY_ENV/bin/activate"

# Install dependencies
pip install --upgrade pip
pip install opencv-python-headless==4.8.1.78
pip install opencv-contrib-python-headless==4.8.1.78
pip install numpy==1.24.3
pip install tflite-runtime==2.14.0 || echo "Note: TFLite runtime installation failed, trying alternative..."
pip install Pillow==10.1.0
pip install scipy==1.11.4

# Try to install FFMPEG Python bindings
pip install ffmpeg-python==0.2.0 || echo "Note: ffmpeg-python installation failed, timelapse may not work"

deactivate
echo "✓ Dependencies installed"

echo ""
echo "Step 4: Creating Vision One directories..."
sudo mkdir -p /tmp/vision_one/{models,calibration,timelapse}
sudo chown -R pi:pi /tmp/vision_one
echo "✓ Directories created"

echo ""
echo "Step 5: Installing system dependencies..."
sudo apt update
sudo apt install -y v4l-utils ffmpeg
echo "✓ System dependencies installed"

echo ""
echo "Step 6: Checking camera..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✓ Camera devices found:"
    ls -l /dev/video*
    
    # Add user to video group
    sudo usermod -a -G video pi
    echo "✓ Added pi user to video group"
else
    echo "⚠ Warning: No camera devices detected"
    echo "  Please connect your USB endoscope and reboot"
fi

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy the configuration from printer.cfg.example to your printer.cfg"
echo "2. Adjust camera_device, resolution, and park positions for your setup"
echo "3. Restart Klipper: sudo systemctl restart klipper"
echo "4. Test with: VISION_STATUS"
echo ""
echo "Optional:"
echo "5. Add AI models to /tmp/vision_one/models/"
echo "6. Reboot to apply video group changes: sudo reboot"
echo ""
