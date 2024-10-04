document.addEventListener("DOMContentLoaded", function () {
    const audioForm = document.getElementById('audioForm');
    const resultDiv = document.getElementById('result');

    audioForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent the default form submission

        const audioInput = document.getElementById('audioInput');
        const audioFile = audioInput.files[0];

        if (!audioFile) {
            resultDiv.textContent = "Please select an audio file.";
            return;
        }

        const formData = new FormData();
        formData.append('audio', audioFile);

        try {
            const response = await fetch('/api/transcribe/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            resultDiv.textContent = "Transcription: " + data.transcription; // Adjust based on your API response structure
        } catch (error) {
            console.error('Error:', error);
            resultDiv.textContent = "Error: " + error.message;
        }
    });
});
