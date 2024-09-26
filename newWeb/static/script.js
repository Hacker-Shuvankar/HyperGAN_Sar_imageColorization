document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageFileInput = document.getElementById('imageFile');
    const noFileMessage = document.getElementById('no-file-message');
    const uploadedImage = document.getElementById('uploadedImage');
    const colorizedImage = document.getElementById('colorizedImage');
    const downloadLink = document.getElementById('downloadLink');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loadingSpinner');
    let uploadedImageURL = null;

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        if (!imageFileInput.files.length) {
            noFileMessage.textContent = 'Please select an image file';
            return;
        }

        noFileMessage.textContent = '';
        errorMessage.textContent = '';
        loadingSpinner.style.display = 'block'; // Show loading spinner

        const formData = new FormData();
        formData.append('file', imageFileInput.files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                errorMessage.textContent = data.error;
            } else {
                // Revoke the previous object URL to prevent memory leaks
                if (uploadedImageURL) {
                    URL.revokeObjectURL(uploadedImageURL);
                }

                // Display the uploaded image
                uploadedImageURL = URL.createObjectURL(imageFileInput.files[0]);
                uploadedImage.src = uploadedImageURL;
                uploadedImage.style.display = 'block';

                // Force cache busting by appending a random query string
                colorizedImage.src = data.colorized_image + '?cache_bust=' + new Date().getTime();
                colorizedImage.style.display = 'block';

                downloadLink.href = data.colorized_image + '?cache_bust=' + new Date().getTime();
                downloadLink.style.display = 'inline';
            }
        })
        .catch(error => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner on error
            errorMessage.textContent = 'An error occurred while processing the image.';
        });
    });

    document.getElementById('cancelButton').addEventListener('click', function() {
        uploadForm.reset();
        uploadedImage.style.display = 'none';
        colorizedImage.style.display = 'none';
        downloadLink.style.display = 'none';
        errorMessage.textContent = '';

        // Revoke the object URL when cancel is clicked
        if (uploadedImageURL) {
            URL.revokeObjectURL(uploadedImageURL);
            uploadedImageURL = null;
        }
    });
});