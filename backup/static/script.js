// JavaScript for Face Recognition System

// Global variables
let isStreaming = false;
let statusInterval = null;

// DOM elements
const startStreamBtn = document.getElementById('start-stream');
const stopStreamBtn = document.getElementById('stop-stream');
const streamImg = document.getElementById('stream');
const streamPlaceholder = document.getElementById('stream-placeholder');
const addFaceForm = document.getElementById('add-face-form');
const subjectNameInput = document.getElementById('subject-name');
const trainModelBtn = document.getElementById('train-model');
const knownSubjectsList = document.getElementById('known-subjects');
const detectionResults = document.getElementById('detection-results');

// Bootstrap modal
let messageModal;
let modalTitle;
let modalBody;

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Bootstrap modal
    messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
    modalTitle = document.getElementById('modalTitle');
    modalBody = document.getElementById('modalBody');
    
    // Get initial status
    updateStatus();
    
    // Set up event listeners
    startStreamBtn.addEventListener('click', startStreaming);
    stopStreamBtn.addEventListener('click', stopStreaming);
    addFaceForm.addEventListener('submit', addFace);
    trainModelBtn.addEventListener('click', trainModel);
});

// Update status periodically when streaming
function setupStatusPolling(enable) {
    if (enable) {
        // Clear any existing interval
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Set up polling every 1 second
        statusInterval = setInterval(updateStatus, 1000);
    } else {
        // Clear interval
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
    }
}

// Update system status
function updateStatus() {
    fetch('/api/get_status')
        .then(response => response.json())
        .then(data => {
            // Update streaming status
            isStreaming = data.is_streaming;
            updateStreamingUI(isStreaming);
            
            // Update known subjects
            if (data.model_loaded) {
                updateKnownSubjects(data.known_subjects);
            } else {
                knownSubjectsList.innerHTML = '<li class="list-group-item">No trained model loaded</li>';
            }
            
            // Show system error if any
            if (data.system_status && data.system_status.error) {
                const errorMessage = document.querySelector('.alert-warning');
                if (!errorMessage) {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-warning';
                    alertDiv.innerHTML = `
                        <strong>System Status Warning:</strong> ${data.system_status.message}
                        <p>The system will operate with limited functionality. OpenCV fallback methods will be used where possible.</p>
                    `;
                    document.querySelector('.container').insertBefore(
                        alertDiv, 
                        document.querySelector('.container').firstChild
                    );
                }
            }
            
            // Update detection results
            updateDetectionResults(data.detection_results);
            
            // Enable/disable add face button
            const addFaceButton = addFaceForm.querySelector('button[type="submit"]');
            addFaceButton.disabled = !isStreaming;
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Update streaming UI elements
function updateStreamingUI(streaming) {
    if (streaming) {
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;
        streamImg.style.display = 'block';
        streamPlaceholder.style.display = 'none';
    } else {
        startStreamBtn.disabled = false;
        stopStreamBtn.disabled = true;
        streamImg.style.display = 'none';
        streamPlaceholder.style.display = 'flex';
    }
}

// Update known subjects list
function updateKnownSubjects(subjects) {
    if (!subjects || subjects.length === 0) {
        knownSubjectsList.innerHTML = '<li class="list-group-item">No subjects in database</li>';
        return;
    }
    
    knownSubjectsList.innerHTML = '';
    subjects.forEach(subject => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        li.textContent = subject;
        
        // Add delete button
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-subject';
        deleteBtn.innerHTML = '<i class="bi bi-trash"></i> Delete';
        deleteBtn.addEventListener('click', () => {
            deleteSubject(subject);
        });
        
        li.appendChild(deleteBtn);
        knownSubjectsList.appendChild(li);
    });
}

// Update detection results
function updateDetectionResults(results) {
    if (!results || results.length === 0) {
        detectionResults.innerHTML = '<p>No faces detected</p>';
        return;
    }
    
    detectionResults.innerHTML = '';
    results.forEach(result => {
        const div = document.createElement('div');
        div.className = 'face-item';
        
        let label = result.label || 'Unknown';
        let confidence = result.confidence || 0;
        
        // Create confidence badge
        let confidenceClass = 'confidence-low';
        if (confidence > 0.8) {
            confidenceClass = 'confidence-high';
        } else if (confidence > 0.6) {
            confidenceClass = 'confidence-medium';
        }
        
        div.innerHTML = `
            <div>
                <span class="confidence ${confidenceClass}">${(confidence * 100).toFixed(0)}%</span>
                <strong>${label}</strong>
            </div>
        `;
        
        detectionResults.appendChild(div);
    });
}

// Start video streaming
function startStreaming() {
    fetch('/api/start_stream', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            updateStatus();
            setupStatusPolling(true);
            
            // Force reload the image src to start streaming
            streamImg.src = '/video_feed?' + new Date().getTime();
        } else {
            showMessage('Error', data.message);
        }
    })
    .catch(error => {
        console.error('Error starting stream:', error);
        showMessage('Error', 'Failed to start streaming');
    });
}

// Stop video streaming
function stopStreaming() {
    fetch('/api/stop_stream', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            updateStatus();
            setupStatusPolling(false);
        } else {
            showMessage('Error', data.message);
        }
    })
    .catch(error => {
        console.error('Error stopping stream:', error);
        showMessage('Error', 'Failed to stop streaming');
    });
}

// Add new face
function addFace(event) {
    event.preventDefault();
    
    const subjectName = subjectNameInput.value.trim();
    if (!subjectName) {
        showMessage('Error', 'Subject name is required');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('subject_name', subjectName);
    
    fetch('/api/add_face', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage('Success', `${data.message} (${data.face_count} images total)`);
            subjectNameInput.value = '';
            updateStatus();
        } else {
            showMessage('Error', data.message);
        }
    })
    .catch(error => {
        console.error('Error adding face:', error);
        showMessage('Error', 'Failed to add face');
    });
}

// Train recognition model
function trainModel() {
    // Confirm
    if (!confirm('This will train the recognition model with all faces in the database. Continue?')) {
        return;
    }
    
    // Disable button
    trainModelBtn.disabled = true;
    trainModelBtn.textContent = 'Training...';
    
    fetch('/api/train_model', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage('Success', data.message);
            updateStatus();
        } else {
            showMessage('Error', data.message);
        }
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = 'Train Recognition Model';
    })
    .catch(error => {
        console.error('Error training model:', error);
        showMessage('Error', 'Failed to train model');
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = 'Train Recognition Model';
    });
}

// Delete subject
function deleteSubject(subject) {
    // Confirm
    if (!confirm(`Are you sure you want to delete ${subject}?`)) {
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('subject_name', subject);
    
    fetch('/api/delete_subject', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage('Success', data.message);
            updateStatus();
        } else {
            showMessage('Error', data.message);
        }
    })
    .catch(error => {
        console.error('Error deleting subject:', error);
        showMessage('Error', 'Failed to delete subject');
    });
}

// Show modal with message
function showMessage(title, message) {
    modalTitle.textContent = title;
    modalBody.textContent = message;
    messageModal.show();
} 