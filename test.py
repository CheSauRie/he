# Video Upscaler Service using OpenCV and FFmpeg
# Required installations:
# pip install flask opencv-python numpy
# FFmpeg must be installed on your system

import os
import uuid
import time
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory
from threading import Thread
from queue import Queue
import cv2
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_QUEUE_SIZE = 10

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Job queue and status tracking
job_queue = Queue(maxsize=MAX_QUEUE_SIZE)
job_status = {}  # Dictionary to track job status

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check if FFmpeg is installed
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Worker function for processing videos
def process_worker():
    while True:
        job_id, input_path, output_path, target_resolution, target_fps = job_queue.get()
        try:
            job_status[job_id]['status'] = 'processing'
            
            # Update status to show progress
            job_status[job_id]['progress'] = 10
            
            # Extract video information
            video = cv2.VideoCapture(input_path)
            original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            video.release()  # Close video
            
            # Parse target resolution
            if target_resolution == '720p':
                target_height = 720
            elif target_resolution == '1080p':
                target_height = 1080
            elif target_resolution == '4k' or target_resolution == '4K':
                target_height = 2160
            else:
                # Default to 720p if invalid
                target_height = 720
                
            # Calculate width maintaining aspect ratio
            target_width = int((target_height / original_height) * original_width)
            # Ensure even dimensions (required by some codecs)
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
            
            # Update status
            job_status[job_id]['progress'] = 20
            
            # Process video using FFmpeg
            if check_ffmpeg():
                # Use FFmpeg for better quality and frame interpolation
                upscale_with_ffmpeg(input_path, output_path, target_width, target_height, 
                                    original_fps, target_fps, job_id)
            else:
                # Fallback to OpenCV
                upscale_with_opencv(input_path, output_path, target_width, target_height, 
                                    original_fps, job_id)
            
            job_status[job_id]['status'] = 'completed'
            job_status[job_id]['progress'] = 100
            
        except Exception as e:
            job_status[job_id]['status'] = 'failed'
            job_status[job_id]['error'] = str(e)
            print(f"Error processing job {job_id}: {str(e)}")
        finally:
            job_queue.task_done()

# FFmpeg-based upscaling with frame interpolation
def upscale_with_ffmpeg(input_path, output_path, target_width, target_height, 
                        original_fps, target_fps, job_id):
    # First pass - upscale the video without changing the framerate
    temp_upscaled = os.path.join(TEMP_FOLDER, f"{job_id}_upscaled_temp.mp4")
    
    # Update progress
    job_status[job_id]['progress'] = 25
    
    # Command for upscaling
    upscale_cmd = [
        'ffmpeg',
        '-i', input_path,                         # Input file
        '-vf', f'scale={target_width}:{target_height}',  # Scale filter
        '-c:v', 'libx264',                        # Video codec
        '-crf', '18',                             # Quality (lower is better)
        '-preset', 'medium',                      # Encoding speed/quality balance
        '-c:a', 'aac',                            # Audio codec
        '-b:a', '128k',                           # Audio bitrate
        '-y',                                     # Overwrite output
        temp_upscaled                             # Output file
    ]
    
    # Run the upscaling command
    process = subprocess.Popen(upscale_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Update progress while processing
    while process.poll() is None:
        time.sleep(1)  # Check status every second
        job_status[job_id]['progress'] = min(50, job_status[job_id]['progress'] + 1)
    
    # Check if upscaling was successful
    if process.returncode != 0:
        stderr = process.stderr.read().decode()
        raise Exception(f"FFmpeg upscaling failed: {stderr}")
    
    # Update progress
    job_status[job_id]['progress'] = 50
    
    # Second pass - apply frame interpolation if target_fps > original_fps
    if target_fps > original_fps:
        # Command for frame interpolation
        interpolate_cmd = [
            'ffmpeg',
            '-i', temp_upscaled,                  # Input file (upscaled)
            '-filter_complex', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',  # Frame interpolation
            '-c:v', 'libx264',                    # Video codec
            '-crf', '18',                         # Quality
            '-preset', 'medium',                  # Encoding speed/quality balance
            '-c:a', 'copy',                       # Copy audio stream
            '-y',                                 # Overwrite output
            output_path                           # Final output file
        ]
        
        # Run the frame interpolation command
        process = subprocess.Popen(interpolate_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Update progress while processing
        while process.poll() is None:
            time.sleep(1)  # Check status every second
            job_status[job_id]['progress'] = min(95, job_status[job_id]['progress'] + 1)
        
        # Check if frame interpolation was successful
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise Exception(f"FFmpeg frame interpolation failed: {stderr}")
    else:
        # If no frame interpolation needed, just copy the upscaled file
        os.rename(temp_upscaled, output_path)
        job_status[job_id]['progress'] = 95
    
    # Clean up temp file if it exists
    if os.path.exists(temp_upscaled):
        os.remove(temp_upscaled)

# OpenCV-based upscaling fallback
def upscale_with_opencv(input_path, output_path, target_width, target_height, original_fps, job_id):
    # Open the video
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (target_width, target_height))
    
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Resize frame to target resolution using Lanczos interpolation for better quality
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Write the frame to output video
        out.write(resized_frame)
        
        count += 1
        
        # Update progress periodically
        if count % 30 == 0 or count == total_frames:
            progress = min(90, 20 + int((count / total_frames) * 70))
            job_status[job_id]['progress'] = progress
    
    # Release resources
    video.release()
    out.release()
    
    # Final processing
    job_status[job_id]['progress'] = 95

# Start the worker thread
worker_thread = Thread(target=process_worker, daemon=True)
worker_thread.start()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get parameters
    target_resolution = request.form.get('resolution', '720p')
    target_fps = int(request.form.get('fps', 60))
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file
    input_filename = f"{job_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)
    
    # Define output path
    output_filename = f"{job_id}_upscaled.mp4"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    # Add job to queue
    if job_queue.full():
        return jsonify({'error': 'Server is busy. Please try again later.'}), 503
        
    # Initialize job status
    job_status[job_id] = {
        'status': 'queued',
        'progress': 0,
        'input_file': input_filename,
        'output_file': output_filename,
        'submitted_at': time.time()
    }
    
    # Add to processing queue
    job_queue.put((job_id, input_path, output_path, target_resolution, target_fps))
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Your video has been queued for processing'
    })

@app.route('/status/<job_id>')
def check_status(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
        
    return jsonify(job_status[job_id])

@app.route('/download/<job_id>')
def download_file(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
        
    job = job_status[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'File not ready for download'}), 400
        
    return send_from_directory(PROCESSED_FOLDER, job['output_file'], as_attachment=True)

# HTML template for the frontend
@app.route('/templates/index.html')
def serve_template():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Upscaler</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .progress-container {
                width: 100%;
                background-color: #f1f1f1;
                border-radius: 5px;
                margin: 10px 0;
            }
            .progress-bar {
                height: 20px;
                background-color: #4CAF50;
                text-align: center;
                line-height: 20px;
                color: white;
                border-radius: 5px;
                transition: width 0.3s;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 10px;
            }
            select, input {
                padding: 8px;
                margin: 5px 0;
                width: 100%;
                box-sizing: border-box;
            }
            #jobList {
                margin-top: 30px;
            }
            .job-item {
                border: 1px solid #ddd;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
            .info-text {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Video Upscaler</h1>
        
        <div class="container">
            <h2>Upload Your Video</h2>
            <form id="uploadForm">
                <div>
                    <label for="videoFile">Select Video:</label>
                    <input type="file" id="videoFile" name="file" accept=".mp4,.avi,.mov,.mkv,.webm">
                </div>
                <div>
                    <label for="resolution">Target Resolution:</label>
                    <select id="resolution" name="resolution">
                        <option value="720p">720p</option>
                        <option value="1080p">1080p</option>
                        <option value="4k">4K</option>
                    </select>
                </div>
                <div>
                    <label for="fps">Target FPS:</label>
                    <select id="fps" name="fps">
                        <option value="30">30 FPS</option>
                        <option value="60" selected>60 FPS</option>
                    </select>
                    <p class="info-text">Higher FPS will create smoother video with frame interpolation.</p>
                </div>
                <button type="submit">Upload & Process</button>
            </form>
        </div>
        
        <div id="jobList">
            <h2>Your Jobs</h2>
            <div id="jobItems"></div>
        </div>
        
        <script>
            // Store active jobs
            const activeJobs = {};
            
            // Check if FFmpeg is available
            fetch('/status/check_ffmpeg')
              .then(response => response.json())
              .catch(() => {
                console.log("Could not verify FFmpeg availability");
              });
            
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const fileInput = document.getElementById('videoFile');
                
                if (!fileInput.files[0]) {
                    alert('Please select a video file');
                    return;
                }
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        // Add job to tracking
                        activeJobs[result.job_id] = {
                            id: result.job_id,
                            status: 'queued',
                            progress: 0,
                            filename: fileInput.files[0].name
                        };
                        
                        // Add to UI
                        addJobToUI(result.job_id, fileInput.files[0].name);
                        
                        // Start polling for status
                        startPolling(result.job_id);
                        
                        // Reset form
                        document.getElementById('uploadForm').reset();
                    } else {
                        alert('Error: ' + result.error);
                    }
                } catch (error) {
                    alert('Upload failed: ' + error.message);
                }
            });
            
            function addJobToUI(jobId, filename) {
                const jobItems = document.getElementById('jobItems');
                
                const jobElement = document.createElement('div');
                jobElement.className = 'job-item';
                jobElement.id = `job-${jobId}`;
                
                jobElement.innerHTML = `
                    <h3>${filename}</h3>
                    <p>Status: <span id="status-${jobId}">Queued</span></p>
                    <div class="progress-container">
                        <div class="progress-bar" id="progress-${jobId}" style="width:0%">0%</div>
                    </div>
                    <div id="download-${jobId}" style="display:none">
                        <a href="/download/${jobId}" target="_blank">
                            <button>Download Upscaled Video</button>
                        </a>
                    </div>
                `;
                
                jobItems.prepend(jobElement);
            }
            
            function updateJobStatus(jobId, status, progress) {
                document.getElementById(`status-${jobId}`).textContent = 
                    status.charAt(0).toUpperCase() + status.slice(1);
                    
                const progressBar = document.getElementById(`progress-${jobId}`);
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${progress}%`;
                
                if (status === 'completed') {
                    document.getElementById(`download-${jobId}`).style.display = 'block';
                }
            }
            
            async function startPolling(jobId) {
                const pollInterval = setInterval(async () => {
                    try {
                        const response = await fetch(`/status/${jobId}`);
                        const jobInfo = await response.json();
                        
                        if (response.ok) {
                            updateJobStatus(jobId, jobInfo.status, jobInfo.progress);
                            
                            // Stop polling when job is complete or failed
                            if (jobInfo.status === 'completed' || jobInfo.status === 'failed') {
                                clearInterval(pollInterval);
                            }
                        } else {
                            console.error('Error polling job status', jobInfo);
                        }
                    } catch (error) {
                        console.error('Error polling job status', error);
                    }
                }, 2000); // Poll every 2 seconds
            }
        </script>
    </body>
    </html>
    """
    return html_content

# Endpoint to check if FFmpeg is available
@app.route('/status/check_ffmpeg')
def ffmpeg_status():
    ffmpeg_available = check_ffmpeg()
    return jsonify({
        'ffmpeg_available': ffmpeg_available,
        'message': 'FFmpeg is available' if ffmpeg_available else 'FFmpeg is not installed. Using OpenCV fallback.'
    })

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)