import pytest
import requests
from io import BytesIO

# Define the base URL for the API
BASE_URL = "http://localhost:8000"  # Change this to your Azure URL

@pytest.fixture
def sample_image():
    """
    Provides a sample image file as a BytesIO object for testing.
    """
    # Replace with the path to a local test image or dynamically generate an image
    image_path = "test_data/sample_image.jpg"
    with open(image_path, "rb") as img:
        yield BytesIO(img.read())

def test_predict_api(sample_image):
    """
    Tests the /predict endpoint of the inference API.
    """
    url = f"{BASE_URL}/predict/"
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}

    # Send the POST request to the /predict API
    response = requests.post(url, files=files)

    # Check if the response status code is 200
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    # Parse the response JSON
    json_response = response.json()

    # Check if the response has predictions
    assert "predictions" in json_response, "No 'predictions' key in response"
    predictions = json_response["predictions"]

    # Ensure predictions are a list
    assert isinstance(predictions, list), f"Predictions is not a list: {type(predictions)}"

    # Validate the structure of each prediction
    for prediction in predictions:
        assert "box" in prediction, "Missing 'box' key in prediction"
        assert "label" in prediction, "Missing 'label' key in prediction"
        assert "score" in prediction, "Missing 'score' key in prediction"
        assert isinstance(prediction["box"], list), "'box' is not a list"
        assert len(prediction["box"]) == 4, "'box' does not contain 4 values"
        assert isinstance(prediction["label"], int), "'label' is not an integer"
        assert isinstance(prediction["score"], float), "'score' is not a float"

    print("Test passed: Predictions are valid.")

if __name__ == "__main__":
    pytest.main()
