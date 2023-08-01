document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();

    var fileInput = document.getElementById('image-input');
    var file = fileInput.files[0];
    var formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(function(response) {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Error occurred while making a request.');
        }
    })
    .then(function(data) {
        var result = data.result;
        displayResult(result);
    })
    .catch(function(error) {
        alert(error.message);
    });
});

function displayResult(result) {
    var resultElement = document.getElementById('result');
    resultElement.textContent = 'Predicted pose: ' + result;
    resultElement.style.display = 'block';
}
