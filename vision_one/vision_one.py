# Klipper Vision One - Advanced Computer Vision & AI Plugin
# Copyright (C) 2026
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging
import threading
import queue
import time
import os
import json
from pathlib import Path

# Computer Vision
import cv2
import numpy as np

# TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    logging.warning("TFLite runtime not available. AI features will be disabled.")
    tflite = None

# Video Processing
try:
    import ffmpeg
except ImportError:
    logging.warning("FFMPEG not available. Timelapse features will be limited.")
    ffmpeg = None


class VisionOne:
    """
    Klipper Vision One Plugin
    Provides computer vision calibration, AI-based monitoring, and timelapse capabilities
    """
    
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name()
        self.gcode = self.printer.lookup_object('gcode')
        
        # Configuration
        self.camera_device = config.getint('camera_device', 0)
        self.camera_width = config.getint('camera_width', 1920)
        self.camera_height = config.getint('camera_height', 1080)
        self.camera_fps = config.getint('camera_fps', 30)
        
        # Paths
        self.base_path = Path(config.get('base_path', '/tmp/vision_one'))
        self.models_path = self.base_path / 'models'
        self.calibration_path = self.base_path / 'calibration'
        self.timelapse_path = self.base_path / 'timelapse'
        
        # Ensure directories exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.calibration_path.mkdir(parents=True, exist_ok=True)
        self.timelapse_path.mkdir(parents=True, exist_ok=True)
        
        # AI Model Configuration
        self.spaghetti_model_path = config.get('spaghetti_model', 
                                                str(self.models_path / 'spaghetti_detector.tflite'))
        self.first_layer_model_path = config.get('first_layer_model',
                                                  str(self.models_path / 'first_layer_classifier.tflite'))
        
        # Feature Flags
        self.enable_spaghetti_detection = config.getboolean('enable_spaghetti_detection', False)
        self.enable_first_layer_inspection = config.getboolean('enable_first_layer_inspection', False)
        self.enable_clog_detection = config.getboolean('enable_clog_detection', False)
        self.enable_timelapse = config.getboolean('enable_timelapse', False)
        
        # Detection Parameters
        self.spaghetti_interval = config.getfloat('spaghetti_check_interval', 10.0)
        self.spaghetti_threshold = config.getfloat('spaghetti_threshold', 0.7)
        self.clog_flow_threshold = config.getfloat('clog_flow_threshold', 2.0)
        
        # Timelapse Configuration
        self.timelapse_park_x = config.getfloat('timelapse_park_x', None)
        self.timelapse_park_y = config.getfloat('timelapse_park_y', None)
        self.timelapse_park_z_lift = config.getfloat('timelapse_park_z_lift', 0.5)
        
        # Camera & Threading
        self.camera = None
        self.camera_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.camera_thread = None
        self.ai_monitor_thread = None
        
        # AI Models
        self.spaghetti_interpreter = None
        self.first_layer_interpreter = None
        
        # State
        self.print_started = False
        self.first_layer_complete = False
        self.last_frame = None
        self.timelapse_frames = []
        self.current_timelapse_file = None
        
        # Register G-Code Commands
        self.gcode.register_command('CALIBRATE_SKEW_VISION',
                                   self.cmd_CALIBRATE_SKEW_VISION,
                                   desc=self.cmd_CALIBRATE_SKEW_VISION_help)
        
        self.gcode.register_command('CALIBRATE_NOZZLE_VISION',
                                   self.cmd_CALIBRATE_NOZZLE_VISION,
                                   desc=self.cmd_CALIBRATE_NOZZLE_VISION_help)
        
        self.gcode.register_command('CALIBRATE_FLOW_VISION',
                                   self.cmd_CALIBRATE_FLOW_VISION,
                                   desc=self.cmd_CALIBRATE_FLOW_VISION_help)
        
        self.gcode.register_command('VISUAL_Z_OFFSET',
                                   self.cmd_VISUAL_Z_OFFSET,
                                   desc=self.cmd_VISUAL_Z_OFFSET_help)
        
        self.gcode.register_command('DETECT_SPAGHETTI',
                                   self.cmd_DETECT_SPAGHETTI,
                                   desc=self.cmd_DETECT_SPAGHETTI_help)
        
        self.gcode.register_command('INSPECT_FIRST_LAYER',
                                   self.cmd_INSPECT_FIRST_LAYER,
                                   desc=self.cmd_INSPECT_FIRST_LAYER_help)
        
        self.gcode.register_command('START_TIMELAPSE',
                                   self.cmd_START_TIMELAPSE,
                                   desc=self.cmd_START_TIMELAPSE_help)
        
        self.gcode.register_command('STOP_TIMELAPSE',
                                   self.cmd_STOP_TIMELAPSE,
                                   desc=self.cmd_STOP_TIMELAPSE_help)
        
        self.gcode.register_command('VISION_STATUS',
                                   self.cmd_VISION_STATUS,
                                   desc=self.cmd_VISION_STATUS_help)
        
        # Register event handlers
        self.printer.register_event_handler("klippy:ready", self.handle_ready)
        self.printer.register_event_handler("klippy:shutdown", self.handle_shutdown)
    
    def handle_ready(self):
        """Initialize camera and AI models when Klipper is ready"""
        reactor = self.printer.get_reactor()
        reactor.register_callback(self._async_initialize)
    
    def _async_initialize(self, eventtime):
        """Asynchronous initialization to avoid blocking Klipper"""
        try:
            self._initialize_camera()
            self._load_ai_models()
            self.running = True
            self._start_background_threads()
            logging.info("Vision One initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Vision One: {e}")
    
    def _initialize_camera(self):
        """Initialize USB camera in non-blocking manner"""
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to read from camera")
            
            logging.info(f"Camera initialized: {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            self.camera = None
    
    def _load_ai_models(self):
        """Load TensorFlow Lite models"""
        if tflite is None:
            logging.warning("TFLite not available, skipping model loading")
            return
        
        try:
            # Load Spaghetti Detection Model
            if self.enable_spaghetti_detection and os.path.exists(self.spaghetti_model_path):
                self.spaghetti_interpreter = tflite.Interpreter(model_path=self.spaghetti_model_path)
                self.spaghetti_interpreter.allocate_tensors()
                logging.info("Spaghetti detection model loaded")
            
            # Load First Layer Inspection Model
            if self.enable_first_layer_inspection and os.path.exists(self.first_layer_model_path):
                self.first_layer_interpreter = tflite.Interpreter(model_path=self.first_layer_model_path)
                self.first_layer_interpreter.allocate_tensors()
                logging.info("First layer inspection model loaded")
        except Exception as e:
            logging.error(f"Failed to load AI models: {e}")
    
    def _start_background_threads(self):
        """Start background threads for camera capture and AI monitoring"""
        # Camera capture thread
        self.camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.camera_thread.start()
        
        # AI monitoring thread (if enabled)
        if self.enable_spaghetti_detection or self.enable_clog_detection:
            self.ai_monitor_thread = threading.Thread(target=self._ai_monitor_loop, daemon=True)
            self.ai_monitor_thread.start()
    
    def _camera_capture_loop(self):
        """Background thread to continuously capture frames"""
        while self.running and self.camera is not None:
            try:
                with self.camera_lock:
                    ret, frame = self.camera.read()
                
                if ret:
                    self.last_frame = frame.copy()
                    
                    # Add to queue if not full (non-blocking)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame, block=False)
                else:
                    logging.warning("Failed to capture frame")
                
                time.sleep(1.0 / self.camera_fps)
            except Exception as e:
                logging.error(f"Camera capture error: {e}")
                time.sleep(1.0)
    
    def _ai_monitor_loop(self):
        """Background thread for AI monitoring tasks"""
        last_spaghetti_check = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Spaghetti Detection (every N seconds)
                if (self.enable_spaghetti_detection and 
                    self.print_started and 
                    current_time - last_spaghetti_check >= self.spaghetti_interval):
                    
                    self._check_spaghetti()
                    last_spaghetti_check = current_time
                
                # Clog Detection (continuous monitoring)
                if self.enable_clog_detection and self.print_started:
                    self._check_clog()
                
                time.sleep(1.0)
            except Exception as e:
                logging.error(f"AI monitor error: {e}")
                time.sleep(1.0)
    
    def _get_latest_frame(self):
        """Get the most recent camera frame"""
        if self.last_frame is not None:
            return self.last_frame.copy()
        return None
    
    # =====================================================================
    # FEATURE 1: Deterministic Computer Vision (OpenCV/ArUco)
    # =====================================================================
    
    cmd_CALIBRATE_SKEW_VISION_help = "Detect ChArUco board and calculate skew correction"
    def cmd_CALIBRATE_SKEW_VISION(self, gcmd):
        """Automated Skew Correction using ChArUco board detection"""
        self.gcode.respond_info("Starting visual skew calibration...")
        
        try:
            frame = self._get_latest_frame()
            if frame is None:
                raise RuntimeError("No camera frame available")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Define ChArUco board parameters
            SQUARES_X = 7
            SQUARES_Y = 5
            SQUARE_LENGTH = 40  # mm
            MARKER_LENGTH = 30  # mm
            
            # Create ChArUco board
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
            
            # Detect markers
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is None or len(ids) < 4:
                raise RuntimeError("Not enough ArUco markers detected. Ensure ChArUco board is visible.")
            
            # Interpolate ChArUco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            
            if not retval or charuco_corners is None or len(charuco_corners) < 4:
                raise RuntimeError("Failed to detect ChArUco corners")
            
            # Calculate skew from corner positions
            skew_angle, scale_x, scale_y = self._calculate_skew_from_corners(charuco_corners, charuco_ids, board)
            
            self.gcode.respond_info(f"Skew detected: {skew_angle:.6f} radians")
            self.gcode.respond_info(f"X scale factor: {scale_x:.6f}")
            self.gcode.respond_info(f"Y scale factor: {scale_y:.6f}")
            self.gcode.respond_info(f"")
            self.gcode.respond_info(f"Add to printer.cfg:")
            self.gcode.respond_info(f"[skew_correction]")
            self.gcode.respond_info(f"# Generated by CALIBRATE_SKEW_VISION")
            self.gcode.respond_info(f"skew_xy: {skew_angle:.6f}")
            
            # Save calibration data
            calibration_data = {
                'timestamp': time.time(),
                'skew_xy': float(skew_angle),
                'scale_x': float(scale_x),
                'scale_y': float(scale_y)
            }
            self._save_calibration('skew_calibration.json', calibration_data)
            
        except Exception as e:
            self.gcode.respond_error(f"Skew calibration failed: {e}")
    
    def _calculate_skew_from_corners(self, corners, ids, board):
        """Calculate skew angle and scale factors from ChArUco corners"""
        # Extract corner positions
        points = corners.reshape(-1, 2)
        
        # Find horizontal and vertical lines
        # This is a simplified calculation - in production, use full calibration matrix
        
        # Calculate angle of best-fit line through all points
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy[0], vx[0])
        
        # Calculate scale factors (simplified)
        x_extent = np.ptp(points[:, 0])
        y_extent = np.ptp(points[:, 1])
        scale_x = 1.0
        scale_y = y_extent / x_extent if x_extent > 0 else 1.0
        
        return angle, scale_x, scale_y
    
    cmd_CALIBRATE_NOZZLE_VISION_help = "Detect reference marker to calculate nozzle offset"
    def cmd_CALIBRATE_NOZZLE_VISION(self, gcmd):
        """Visual Nozzle Alignment using circular marker detection"""
        self.gcode.respond_info("Starting visual nozzle alignment...")
        
        try:
            frame = self._get_latest_frame()
            if frame is None:
                raise RuntimeError("No camera frame available")
            
            # Get current toolhead position
            toolhead = self.printer.lookup_object('toolhead')
            current_pos = toolhead.get_position()
            
            # Convert to grayscale and detect circles
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                      param1=50, param2=30, minRadius=10, maxRadius=100)
            
            if circles is None:
                raise RuntimeError("No circular reference marker detected")
            
            circles = np.uint16(np.around(circles))
            
            # Use the most prominent circle (first one)
            x, y, r = circles[0][0]
            
            # Calculate pixel-to-mm conversion (simplified - needs camera calibration)
            # Assuming camera FOV covers known bed area
            CAMERA_FOV_MM_X = 100.0  # Example: camera sees 100mm in X
            px_to_mm = CAMERA_FOV_MM_X / self.camera_width
            
            # Calculate offset from center
            center_x = self.camera_width / 2
            center_y = self.camera_height / 2
            
            offset_x = (x - center_x) * px_to_mm
            offset_y = (center_y - y) * px_to_mm  # Invert Y axis
            
            self.gcode.respond_info(f"Detected marker at pixel: ({x}, {y})")
            self.gcode.respond_info(f"Calculated nozzle offset: X={offset_x:.3f}mm, Y={offset_y:.3f}mm")
            self.gcode.respond_info(f"Current toolhead position: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}")
            self.gcode.respond_info(f"")
            self.gcode.respond_info(f"Suggested absolute nozzle position:")
            self.gcode.respond_info(f"X={current_pos[0] + offset_x:.3f}, Y={current_pos[1] + offset_y:.3f}")
            
        except Exception as e:
            self.gcode.respond_error(f"Nozzle alignment failed: {e}")
    
    cmd_CALIBRATE_FLOW_VISION_help = "Analyze test pattern to suggest extrusion multiplier"
    def cmd_CALIBRATE_FLOW_VISION(self, gcmd):
        """Flow Rate Calibration using printed line width analysis"""
        self.gcode.respond_info("Starting visual flow rate calibration...")
        
        try:
            frame = self._get_latest_frame()
            if frame is None:
                raise RuntimeError("No camera frame available")
            
            # Expected line width from G-code
            expected_width_mm = gcmd.get_float('EXPECTED_WIDTH', 0.4)
            
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise RuntimeError("No printed lines detected in image")
            
            # Analyze largest contour (assumed to be the calibration line)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate actual width in pixels (use minimum dimension)
            actual_width_px = min(w, h)
            
            # Convert to mm (using same px_to_mm ratio from camera calibration)
            CAMERA_FOV_MM_X = 100.0
            px_to_mm = CAMERA_FOV_MM_X / self.camera_width
            actual_width_mm = actual_width_px * px_to_mm
            
            # Calculate flow multiplier adjustment
            current_multiplier = 1.0  # Get from printer config if available
            suggested_multiplier = current_multiplier * (expected_width_mm / actual_width_mm)
            
            self.gcode.respond_info(f"Expected line width: {expected_width_mm:.3f}mm")
            self.gcode.respond_info(f"Measured line width: {actual_width_mm:.3f}mm ({actual_width_px}px)")
            self.gcode.respond_info(f"Suggested extrusion multiplier: {suggested_multiplier:.4f}")
            self.gcode.respond_info(f"")
            self.gcode.respond_info(f"Add to printer.cfg:")
            self.gcode.respond_info(f"[extruder]")
            self.gcode.respond_info(f"rotation_distance: <adjust based on multiplier>")
            
        except Exception as e:
            self.gcode.respond_error(f"Flow calibration failed: {e}")
    
    cmd_VISUAL_Z_OFFSET_help = "Display camera feed with crosshair overlay for Z-offset calibration"
    def cmd_VISUAL_Z_OFFSET(self, gcmd):
        """Visual Z-Offset calibration with live crosshair overlay"""
        self.gcode.respond_info("Starting visual Z-offset calibration...")
        self.gcode.respond_info("Press Ctrl+C in console to stop streaming")
        
        try:
            duration = gcmd.get_float('DURATION', 30.0)
            start_time = time.time()
            
            # Create output directory for frames
            output_dir = self.calibration_path / 'z_offset'
            output_dir.mkdir(exist_ok=True)
            
            frame_count = 0
            
            while time.time() - start_time < duration:
                frame = self._get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Draw crosshair overlay
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Draw crosshair
                cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 2)
                cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add text overlay
                cv2.putText(frame, "Z-Offset Calibration", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save frame
                frame_path = output_dir / f'frame_{frame_count:04d}.jpg'
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
                
                time.sleep(0.1)
            
            self.gcode.respond_info(f"Captured {frame_count} frames to {output_dir}")
            self.gcode.respond_info("Review frames and adjust Z-offset accordingly")
            
        except Exception as e:
            self.gcode.respond_error(f"Visual Z-offset failed: {e}")
    
    # =====================================================================
    # FEATURE 2: AI & Machine Learning (TFLite)
    # =====================================================================
    
    cmd_DETECT_SPAGHETTI_help = "Run spaghetti detection on current print"
    def cmd_DETECT_SPAGHETTI(self, gcmd):
        """Manually trigger spaghetti detection"""
        self.gcode.respond_info("Running spaghetti detection...")
        
        try:
            result = self._check_spaghetti()
            
            if result['detected']:
                self.gcode.respond_info(f"⚠️ SPAGHETTI DETECTED! Confidence: {result['confidence']:.2%}")
                
                # Pause print if requested
                if gcmd.get_int('PAUSE_ON_DETECT', 1):
                    self.gcode.run_script_from_command("PAUSE")
                    self.gcode.respond_info("Print paused due to spaghetti detection")
            else:
                self.gcode.respond_info(f"✓ No spaghetti detected. Confidence: {result['confidence']:.2%}")
                
        except Exception as e:
            self.gcode.respond_error(f"Spaghetti detection failed: {e}")
    
    def _check_spaghetti(self):
        """Internal method to check for spaghetti using TFLite model"""
        if self.spaghetti_interpreter is None:
            logging.warning("Spaghetti detection model not loaded")
            return {'detected': False, 'confidence': 0.0}
        
        frame = self._get_latest_frame()
        if frame is None:
            return {'detected': False, 'confidence': 0.0}
        
        try:
            # Preprocess frame for model
            input_details = self.spaghetti_interpreter.get_input_details()
            output_details = self.spaghetti_interpreter.get_output_details()
            
            input_shape = input_details[0]['shape']
            input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
            
            # Run inference
            self.spaghetti_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.spaghetti_interpreter.invoke()
            
            # Get prediction
            output_data = self.spaghetti_interpreter.get_tensor(output_details[0]['index'])
            confidence = float(output_data[0][0])
            
            detected = confidence > self.spaghetti_threshold
            
            if detected:
                logging.warning(f"Spaghetti detected! Confidence: {confidence:.2%}")
                self._trigger_spaghetti_alert(confidence)
            
            return {'detected': detected, 'confidence': confidence}
            
        except Exception as e:
            logging.error(f"Spaghetti detection error: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _trigger_spaghetti_alert(self, confidence):
        """Trigger alert and pause print when spaghetti is detected"""
        # Use reactor to safely pause from background thread
        reactor = self.printer.get_reactor()
        reactor.register_callback(lambda et: self._pause_print_spaghetti(confidence))
    
    def _pause_print_spaghetti(self, confidence, eventtime=None):
        """Pause print due to spaghetti detection"""
        self.gcode.respond_info(f"⚠️ SPAGHETTI DETECTED! Confidence: {confidence:.2%}")
        self.gcode.run_script_from_command("PAUSE")
    
    cmd_INSPECT_FIRST_LAYER_help = "Analyze first layer quality using AI"
    def cmd_INSPECT_FIRST_LAYER(self, gcmd):
        """First Layer Inspection using AI classifier"""
        self.gcode.respond_info("Analyzing first layer quality...")
        
        try:
            frame = self._get_latest_frame()
            if frame is None:
                raise RuntimeError("No camera frame available")
            
            if self.first_layer_interpreter is None:
                raise RuntimeError("First layer model not loaded")
            
            # Preprocess frame
            input_details = self.first_layer_interpreter.get_input_details()
            output_details = self.first_layer_interpreter.get_output_details()
            
            input_shape = input_details[0]['shape']
            input_data = cv2.resize(frame, (input_shape[2], input_shape[1]))
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
            
            # Run inference
            self.first_layer_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.first_layer_interpreter.invoke()
            
            # Get prediction (0 = Bad, 1 = Good)
            output_data = self.first_layer_interpreter.get_tensor(output_details[0]['index'])
            quality_score = float(output_data[0][0])
            
            if quality_score > 0.7:
                self.gcode.respond_info(f"✓ First layer quality: GOOD ({quality_score:.2%})")
            elif quality_score > 0.4:
                self.gcode.respond_info(f"⚠️ First layer quality: FAIR ({quality_score:.2%})")
            else:
                self.gcode.respond_info(f"❌ First layer quality: POOR ({quality_score:.2%})")
                self.gcode.respond_info("Consider re-leveling or adjusting Z-offset")
            
        except Exception as e:
            self.gcode.respond_error(f"First layer inspection failed: {e}")
    
    def _check_clog(self):
        """Clog Detection using optical flow analysis"""
        # This would monitor filament/print motion using optical flow
        # Simplified implementation - check if extruder is moving but no visual motion
        
        frame = self._get_latest_frame()
        if frame is None or not hasattr(self, 'prev_frame'):
            self.prev_frame = frame
            return
        
        try:
            # Calculate optical flow
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current, None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude of motion
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = np.mean(magnitude)
            
            # Check if extruder is active (would need to query extruder state)
            # If extruder moving but avg_motion < threshold, potential clog
            
            if avg_motion < self.clog_flow_threshold:
                logging.warning(f"Low optical flow detected: {avg_motion:.2f}")
                # Could trigger alert here
            
            self.prev_frame = frame
            
        except Exception as e:
            logging.error(f"Clog detection error: {e}")
    
    # =====================================================================
    # FEATURE 3: Media Features (Timelapse)
    # =====================================================================
    
    cmd_START_TIMELAPSE_help = "Start timelapse recording"
    def cmd_START_TIMELAPSE(self, gcmd):
        """Start stabilized timelapse recording"""
        self.gcode.respond_info("Starting timelapse...")
        
        try:
            # Initialize timelapse
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_timelapse_file = self.timelapse_path / f"timelapse_{timestamp}.mp4"
            self.timelapse_frames = []
            
            # Hook into layer change events
            self._register_layer_change_hook()
            
            self.gcode.respond_info(f"Timelapse started: {self.current_timelapse_file.name}")
            
        except Exception as e:
            self.gcode.respond_error(f"Failed to start timelapse: {e}")
    
    cmd_STOP_TIMELAPSE_help = "Stop timelapse recording and compile video"
    def cmd_STOP_TIMELAPSE(self, gcmd):
        """Stop timelapse and compile video"""
        self.gcode.respond_info("Stopping timelapse...")
        
        try:
            if not self.timelapse_frames:
                self.gcode.respond_info("No timelapse frames captured")
                return
            
            # Compile video using FFMPEG
            self._compile_timelapse_video()
            
            self.gcode.respond_info(f"Timelapse saved: {self.current_timelapse_file}")
            self.gcode.respond_info(f"Total frames: {len(self.timelapse_frames)}")
            
            # Cleanup
            self.timelapse_frames = []
            self.current_timelapse_file = None
            
        except Exception as e:
            self.gcode.respond_error(f"Failed to stop timelapse: {e}")
    
    def _register_layer_change_hook(self):
        """Hook into G-code to detect layer changes"""
        # This would require hooking into G-code parser
        # For now, demonstrate the concept
        pass
    
    def _capture_timelapse_frame(self):
        """Capture a frame for timelapse (called on layer change)"""
        try:
            # Park toolhead
            if self.timelapse_park_x is not None and self.timelapse_park_y is not None:
                toolhead = self.printer.lookup_object('toolhead')
                current_pos = toolhead.get_position()
                
                # Move to park position
                park_gcode = f"G1 X{self.timelapse_park_x} Y{self.timelapse_park_y} F3000"
                self.gcode.run_script_from_command(park_gcode)
                
                # Wait for move to complete
                toolhead.wait_moves()
                
                # Capture frame
                time.sleep(0.2)  # Allow camera to stabilize
                frame = self._get_latest_frame()
                
                if frame is not None:
                    frame_path = self.timelapse_path / f"frame_{len(self.timelapse_frames):05d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    self.timelapse_frames.append(str(frame_path))
                
                # Return to previous position
                return_gcode = f"G1 X{current_pos[0]} Y{current_pos[1]} F3000"
                self.gcode.run_script_from_command(return_gcode)
                
        except Exception as e:
            logging.error(f"Timelapse frame capture error: {e}")
    
    def _compile_timelapse_video(self):
        """Compile captured frames into video using FFMPEG"""
        if ffmpeg is None or not self.timelapse_frames:
            return
        
        try:
            # Use FFMPEG to create video from frames
            frame_pattern = str(self.timelapse_path / "frame_%05d.jpg")
            
            (
                ffmpeg
                .input(frame_pattern, framerate=30)
                .output(str(self.current_timelapse_file), 
                       vcodec='libx264', 
                       pix_fmt='yuv420p',
                       crf=20)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Cleanup frame files
            for frame_path in self.timelapse_frames:
                try:
                    os.remove(frame_path)
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"Timelapse compilation error: {e}")
    
    # =====================================================================
    # Utility Functions
    # =====================================================================
    
    cmd_VISION_STATUS_help = "Display Vision One status and configuration"
    def cmd_VISION_STATUS(self, gcmd):
        """Display current status of Vision One plugin"""
        status = []
        status.append("=== Vision One Status ===")
        status.append(f"Camera: {'OK' if self.camera is not None else 'NOT AVAILABLE'}")
        status.append(f"Resolution: {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
        status.append(f"")
        status.append("Features:")
        status.append(f"  Spaghetti Detection: {'ENABLED' if self.enable_spaghetti_detection else 'DISABLED'}")
        status.append(f"  First Layer Inspection: {'ENABLED' if self.enable_first_layer_inspection else 'DISABLED'}")
        status.append(f"  Clog Detection: {'ENABLED' if self.enable_clog_detection else 'DISABLED'}")
        status.append(f"  Timelapse: {'ENABLED' if self.enable_timelapse else 'DISABLED'}")
        status.append(f"")
        status.append("AI Models:")
        status.append(f"  Spaghetti Model: {'LOADED' if self.spaghetti_interpreter else 'NOT LOADED'}")
        status.append(f"  First Layer Model: {'LOADED' if self.first_layer_interpreter else 'NOT LOADED'}")
        status.append(f"")
        status.append(f"Base Path: {self.base_path}")
        
        for line in status:
            self.gcode.respond_info(line)
    
    def _save_calibration(self, filename, data):
        """Save calibration data to JSON file"""
        try:
            filepath = self.calibration_path / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Calibration saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save calibration: {e}")
    
    def handle_shutdown(self):
        """Cleanup on Klipper shutdown"""
        self.running = False
        
        # Wait for threads to finish
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=2.0)
        
        if self.ai_monitor_thread is not None:
            self.ai_monitor_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        logging.info("Vision One shutdown complete")


def load_config(config):
    """Klipper plugin entry point"""
    return VisionOne(config)
