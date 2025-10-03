import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore


# Flask Initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
model = load_model('model_inception.h5')

# Class indices 
class_indices = {'Glaucoma': 0, 'Normal': 1}
class_labels = {v: k for k, v in class_indices.items()}

# Home route with upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(img_path)

            # Process image
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            # Predict
            predictions = model.predict(img_data)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_label = class_labels[predicted_class_index]

            # Display result
            if predicted_label.lower() == 'glaucoma':
                prediction = 'Yes - Glaucoma detected'
            else:
                prediction = 'No - Normal eye'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
    