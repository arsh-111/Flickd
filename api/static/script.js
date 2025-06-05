// JavaScript for handling video upload and displaying results

document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default form submission

    const fileInput = document.getElementById('videoFile');
    const resultsDiv = document.getElementById('results');
    const jsonOutputPre = document.getElementById('jsonOutput');

    resultsDiv.style.display = 'block'; // Show results section
    jsonOutputPre.textContent = 'Processing...'; // Show loading message

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    try {
        const response = await fetch('/process-video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            jsonOutputPre.textContent = `Error: ${error.detail || response.statusText}`;
            jsonOutputPre.style.color = 'red';
            return;
        }

        const result = await response.json();
        jsonOutputPre.textContent = JSON.stringify(result, null, 2); // Display formatted JSON
        jsonOutputPre.style.color = '#333'; // Reset color

    } catch (error) {
        jsonOutputPre.textContent = `An error occurred: ${error}`;
        jsonOutputPre.style.color = 'red';
    }
}); 