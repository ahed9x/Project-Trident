# ChArUco Calibration Board Generator
# Use this to generate a calibration board for CALIBRATE_SKEW_VISION

import cv2
import numpy as np
from pathlib import Path

# Board parameters (must match vision_one.py settings)
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 40  # mm
MARKER_LENGTH = 30  # mm

# Output settings
DPI = 300  # Print quality
OUTPUT_DIR = Path(__file__).parent / "calibration"
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_charuco_board():
    """Generate a ChArUco calibration board"""
    
    # Create ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )
    
    # Calculate image size based on DPI
    # Convert mm to inches to pixels
    mm_to_inch = 0.0393701
    board_width_mm = SQUARES_X * SQUARE_LENGTH
    board_height_mm = SQUARES_Y * SQUARE_LENGTH
    
    img_width = int(board_width_mm * mm_to_inch * DPI)
    img_height = int(board_height_mm * mm_to_inch * DPI)
    
    # Generate board image
    img = board.generateImage((img_width, img_height), marginSize=20, borderBits=1)
    
    # Add border and info text
    border_size = 100
    img_with_border = cv2.copyMakeBorder(
        img, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = 50
    cv2.putText(img_with_border, "Klipper Vision One - ChArUco Calibration Board",
                (border_size, text_y), font, 0.8, 0, 2)
    
    text_y += 40
    cv2.putText(img_with_border, f"Squares: {SQUARES_X}x{SQUARES_Y} | Square Size: {SQUARE_LENGTH}mm | Marker Size: {MARKER_LENGTH}mm",
                (border_size, text_y), font, 0.6, 0, 2)
    
    # Add instructions at bottom
    text_y = img_with_border.shape[0] - 60
    cv2.putText(img_with_border, "1. Print this page at 100% scale (no scaling)",
                (border_size, text_y), font, 0.6, 0, 2)
    text_y += 30
    cv2.putText(img_with_border, "2. Attach firmly to a flat surface (glass/aluminum)",
                (border_size, text_y), font, 0.6, 0, 2)
    text_y += 30
    cv2.putText(img_with_border, "3. Place on print bed and run: CALIBRATE_SKEW_VISION",
                (border_size, text_y), font, 0.6, 0, 2)
    
    # Save outputs
    output_png = OUTPUT_DIR / "charuco_board.png"
    output_pdf = OUTPUT_DIR / "charuco_board.pdf"
    
    cv2.imwrite(str(output_png), img_with_border)
    
    print(f"✓ ChArUco board generated successfully!")
    print(f"  PNG: {output_png}")
    print(f"  Board size: {board_width_mm}mm x {board_height_mm}mm")
    print(f"  Image size: {img_with_border.shape[1]}x{img_with_border.shape[0]} pixels")
    print(f"\nPrinting Instructions:")
    print(f"  1. Open {output_png}")
    print(f"  2. Print at 100% scale (NO SCALING)")
    print(f"  3. Use high quality paper or photo paper")
    print(f"  4. Verify printed square size with ruler: {SQUARE_LENGTH}mm")
    print(f"  5. Mount flat on rigid surface")
    
    return img_with_border

def generate_reference_marker():
    """Generate a simple circular reference marker for nozzle alignment"""
    
    # Create white background
    img_size = 800
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255
    
    # Draw concentric circles
    center = (img_size // 2, img_size // 2)
    cv2.circle(img, center, 200, 0, 2)
    cv2.circle(img, center, 150, 0, 2)
    cv2.circle(img, center, 100, 0, 2)
    cv2.circle(img, center, 50, 0, -1)  # Filled center
    
    # Add crosshair
    cv2.line(img, (center[0] - 250, center[1]), (center[0] + 250, center[1]), 0, 2)
    cv2.line(img, (center[0], center[1] - 250), (center[0], center[1] + 250), 0, 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Nozzle Alignment Reference Marker",
                (img_size // 2 - 300, 50), font, 0.8, 0, 2)
    
    cv2.putText(img, "Place at known position on bed",
                (img_size // 2 - 200, img_size - 50), font, 0.6, 0, 2)
    
    # Save
    output_file = OUTPUT_DIR / "reference_marker.png"
    cv2.imwrite(str(output_file), img)
    
    print(f"\n✓ Reference marker generated: {output_file}")
    print(f"  Use this for CALIBRATE_NOZZLE_VISION command")

if __name__ == "__main__":
    print("Klipper Vision One - Calibration Board Generator")
    print("=" * 50)
    print()
    
    try:
        generate_charuco_board()
        generate_reference_marker()
        
        print("\n" + "=" * 50)
        print("Generation complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
