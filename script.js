document.getElementById('analyzeButton').addEventListener('click', function() {
    const text = document.getElementById('textInput').value;

    // Send AJAX request to your backend with the text
    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').textContent = `Sentiment: ${data.sentiment}`;
        });
});