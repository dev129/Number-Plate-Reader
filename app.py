from flask import Flask, render_template, Response, jsonify
import cv2
import easyocr
import re
from datetime import datetime
import sqlite3
import time
import torch

app = Flask(__name__)

# Global variables
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
OPTIMAL_PLATE_AREA_MIN = 15000  # Minimum area for clear plate reading
OPTIMAL_PLATE_AREA_MAX = 50000  # Maximum area for clear plate reading
DETECTION_COOLDOWN = 3  # seconds
FRAME_SKIP = 3  # Process every 3rd frame
CONFIDENCE_THRESHOLD = 0.45

# Initialize camera and CUDA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Initialize EasyOCR with GPU
reader = easyocr.Reader(['en'], gpu=True)

# Global state variables
move_ahead = False
last_detection_time = 0
distance_guidance = "INITIALIZING"

# CUDA stream for parallel processing
cuda_stream = cv2.cuda.Stream()

class DBManager:
    def __init__(self):
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect('practice.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS plates
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     plate_number TEXT UNIQUE,
                     timestamp DATETIME,
                     confidence REAL)''')
        conn.commit()
        conn.close()

    def save_plate(self, plate_number, confidence):
        try:
            conn = sqlite3.connect('practice.db')
            c = conn.cursor()
            
            c.execute("SELECT plate_number FROM plates WHERE plate_number = ?", (plate_number,))
            if not c.fetchone():
                c.execute("""INSERT INTO plates (plate_number, timestamp, confidence) 
                           VALUES (?, ?, ?)""",
                        (plate_number, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), confidence))
                conn.commit()
                success = True
            else:
                success = False
            
            conn.close()
            return success
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False

class PlateProcessor:
    def __init__(self):
        # Initialize CUDA-based image processing elements
        self.gpu_stream = cv2.cuda.Stream()
        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
        
    def preprocess_image(self, frame_gpu):
        # Convert to grayscale on GPU
        gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY, stream=self.gpu_stream)
        
        # Apply Gaussian blur on GPU
        blurred_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5, 5), 0, stream=self.gpu_stream)
        
        # Apply adaptive threshold on GPU
        thresh_gpu = cv2.cuda.threshold(blurred_gpu, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
                                      stream=self.gpu_stream)[1]
        
        return thresh_gpu

    @staticmethod
    def validate_indian_plate(text):
        text = ''.join(text.split()).upper()
        pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
        return bool(re.match(pattern, text))

    def get_distance_guidance(self, contour_area):
        if contour_area < OPTIMAL_PLATE_AREA_MIN:
            return "MOVE CLOSER"
        elif contour_area > OPTIMAL_PLATE_AREA_MAX:
            return "MOVE BACK"
        return "OPTIMAL DISTANCE"

    def process_frame(self, frame):
        global move_ahead, last_detection_time, distance_guidance
        
        # Upload frame to GPU
        frame_gpu = cv2.cuda.GpuMat()
        frame_gpu.upload(frame)
        
        # Preprocess image
        processed_gpu = self.preprocess_image(frame_gpu)
        
        # Download processed image for contour detection (CPU-based)
        processed = processed_gpu.download()
        
        # Find contours (still CPU-based as CUDA doesn't support findContours)
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_plate = None
        best_confidence = 0
        plate_contour = None
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip too small contours
                continue
                
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Looking for rectangular shapes
                # Get distance guidance based on contour area
                distance_guidance = self.get_distance_guidance(area)
                
                # If distance is optimal, try to read the plate
                if distance_guidance == "OPTIMAL DISTANCE":
                    # Extract plate region
                    x, y, w, h = cv2.boundingRect(contour)
                    plate_region = frame[y:y+h, x:x+w]
                    
                    # Use EasyOCR to read the plate
                    results = reader.readtext(plate_region)
                    
                    for (bbox, text, confidence) in results:
                        if confidence > CONFIDENCE_THRESHOLD and self.validate_indian_plate(text):
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_plate = text
                                plate_contour = approx
                
        if best_plate:
            formatted_plate = f"{best_plate[:2]} {best_plate[2:4]} {best_plate[4:6]} {best_plate[6:]}"
            if DBManager().save_plate(formatted_plate, best_confidence):
                move_ahead = True
                last_detection_time = time.time()
                return True, formatted_plate, plate_contour, distance_guidance
                
        return False, None, None, distance_guidance

def generate_frames():
    global move_ahead, last_detection_time
    
    processor = PlateProcessor()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
            
        display_frame = frame.copy()
        current_time = time.time()
        
        # Reset move_ahead after cooldown
        if move_ahead and current_time - last_detection_time > DETECTION_COOLDOWN:
            move_ahead = False
            
        # Process frame if not in move_ahead state
        if not move_ahead:
            success, plate_text, contour, guidance = processor.process_frame(frame)
            
            # Draw distance guidance
            cv2.putText(display_frame, guidance, (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if success:
                # Draw plate number and contour
                cv2.putText(display_frame, f"Plate: {plate_text}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 3)
        
        # Draw move ahead message
        if move_ahead:
            cv2.putText(display_frame, "MOVE AHEAD", 
                       (int(display_frame.shape[1]/2) - 100, int(display_frame.shape[0]/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset')
def reset():
    global move_ahead
    move_ahead = False
    return jsonify({"status": "reset"})

@app.route('/get_status')
def get_status():
    return jsonify({
        "move_ahead": move_ahead,
        "distance_guidance": distance_guidance
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)