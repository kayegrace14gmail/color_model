from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('api/best_model.h5')

class ClassificationResult(BaseModel):
    predicted_class: str
    predicted_probability: float

@csrf_exempt
def classify_image(request):
    if request.method == 'POST':
        # Read and preprocess the image

        image_file = request.FILES['image']
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        IMG_SIZE = 50
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        Xt = new_array
        Xt = np.array(Xt).reshape( IMG_SIZE, IMG_SIZE, 3)

        # Convert image to numpy array and normalize pixel values
        
        Xt = Xt/255

        # Perform inference
        prediction = model.predict(np.expand_dims(Xt, axis=0))

        # Get the predicted class
        predicted_class = np.argmax(prediction)
        predicted_probability = np.max(prediction)

        # Define a dictionary to map the class index to a human-readable label
        class_labels = {
            0: "Dark",
            1: "Green",
            2: "Light",
            3: "Medium",
            # Add more class labels as needed
        }

        # Create the ClassificationResult object for the response
        result = ClassificationResult(
            predicted_class=class_labels[predicted_class],
            predicted_probability=float(predicted_probability)
        )

        # Return the result as a JSON response
        return JsonResponse(result.dict())

    return JsonResponse({'error': 'Invalid request method.'})
