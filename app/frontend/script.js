const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');
const captureBtn = document.getElementById('capture-btn');

let ws;
let isDetecting = false;

// Start camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    statusText.textContent = "Camera ready. Click 'Analyze Emotion' to start real-time detection.";
  } catch (err) {
    statusText.textContent = "Error accessing camera.";
    console.error(err);
  }
}

function initWebSocket() {
  // Connect to the FastAPI WebSocket endpoint
  ws = new WebSocket('ws://localhost:8000/ws/predict_emotion');

  ws.onopen = () => {
    statusText.textContent = "Connected to AI Engine. Analyzing...";
    isDetecting = true;
    sendFrameLoop(); // Start the loop
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    drawPredictions(data.results);
  };

  ws.onclose = () => {
    statusText.textContent = "Disconnected from server.";
    isDetecting = false;
  };
}

function sendFrameLoop() {
  if (!isDetecting || ws.readyState !== WebSocket.OPEN) return;

  // Use a hidden canvas to extract the JPEG data
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = video.videoWidth;
  offscreenCanvas.height = video.videoHeight;
  const offCtx = offscreenCanvas.getContext('2d');
  
  // Draw the current video frame to the hidden canvas
  offCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
  
  // Convert to highly compressed base64 JPEG to save bandwidth (0.5 quality)
  const dataURL = offscreenCanvas.toDataURL('image/jpeg', 0.5);
  
  // Send to backend
  ws.send(dataURL);

  // Wait a brief moment before sending the next frame to prevent flooding the backend
  setTimeout(() => requestAnimationFrame(sendFrameLoop), 100); 
}

function drawPredictions(results) {
  // Clear previous drawings
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!results || results.length === 0) {
    return;
  }

  results.forEach(pred => {
    const [x, y, w, h] = pred.box;
    const emotion = pred.emotion;

    // Draw Box
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Draw Label Background for better visibility
    ctx.fillStyle = 'lime';
    ctx.fillRect(x, y - 25, w, 25);

    // Draw Text
    ctx.fillStyle = 'black';
    ctx.font = 'bold 16px Arial';
    ctx.fillText(emotion, x + 5, y - 7);
  });
}

// Handle button click to start the WebSocket connection
captureBtn.addEventListener('click', () => {
  if (!isDetecting) {
    initWebSocket();
    captureBtn.textContent = "Stop Analysis";
  } else {
    ws.close();
    captureBtn.textContent = "Analyze Emotion";
  }
});

// Initialize
startCamera();