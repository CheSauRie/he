# Simple Video Upscaler Service using only OpenCV
# Required installations:
# pip install flask opencv-python numpy

import os
import uuid
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from threading import Thread
from queue import Queue
import cv2
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_QUEUE_SIZE = 10

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Job queue and status tracking
job_queue = Queue(maxsize=MAX_QUEUE_SIZE)
job_status = {}  # Dictionary to track job status

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()  # Close video to reopen it later
            
            # Parse target resolution
            if target_resolution == '720p':
                target_height = 720
                target_width = int((target_height / original_height) * original_width)
                # Ensure even dimensions (required by some codecs)
                target_width = target_width - (target_width % 2)
            elif target_resolution == '1080p':
                target_height = 1080
                target_width = int((target_height / original_height) * original_width)
                target_width = target_width - (target_width % 2)
            elif target_resolution == '4k':
                target_height = 2160
                target_width = int((target_height / original_height) * original_width)
                target_width = target_width - (target_width % 2)
            else:
                # Default to 720p if invalid
                target_height = 720
                target_width = int((target_height / original_height) * original_width)
                target_width = target_width - (target_width % 2)
            
            # Update status
            job_status[job_id]['progress'] = 20
            
            # Process video using OpenCV for basic upscaling
            upscale_with_opencv(input_path, output_path, target_width, target_height, target_fps, job_id)
            
            job_status[job_id]['status'] = 'completed'
            job_status[job_id]['progress'] = 100
            
        except Exception as e:
            job_status[job_id]['status'] = 'failed'
            job_status[job_id]['error'] = str(e)
            print(f"Error processing job {job_id}: {str(e)}")
        finally:
            job_queue.task_done()

# OpenCV-based upscaling
def upscale_with_opencv(input_path, output_path, target_width, target_height, target_fps, job_id):
    # Open the video
    video = cv2.VideoCapture(input_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set the target FPS
    output_fps = target_fps if target_fps else original_fps
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (target_width, target_height))
    
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
    
    # Final processing (convert to mp4 with H.264 codec if needed)
    job_status[job_id]['progress'] = 95
    
    # If system has ffmpeg installed, we could use it here for better quality encoding
    # But we'll keep it simple for now with just OpenCV

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

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)