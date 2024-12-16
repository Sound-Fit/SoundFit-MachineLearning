# app.py
import base64
import random
from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import joblib
from firebase_admin import credentials, initialize_app
import os

# Firebase Admin SDK Initialization
cred = credentials.Certificate('config/firebase-credentials.json')  # Adjust the path to your Firebase credentials
initialize_app(cred, {'storageBucket': 'soundfit-bfedd.appspot.com'})

app = Flask(__name__)

# Load model
model = joblib.load('assets/nn_20_canny_model_0.58.pkl')  # Adjust to your model path

# Helper functions
def download_image(image_url):
    """Download image from a URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def face_detection(image):
    """Detect face and crop the first detected face."""
    try:
        cascade_path = "assets/haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Haarcascade file not found.")
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Pilih berdasarkan area (w * h)
            x, y, w, h = largest_face
            
            # Memotong gambar sesuai area deteksi wajah terbesar
            face_crop = image[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))

            # Mengurangi area gambar sebesar 10% (90% dari ukuran asli)
            height, width = face_crop.shape[:2]
            new_height = int(height * 0.70)
            new_width = int(width * 0.70)

            # Menghitung margin untuk cropping agar tetap di tengah
            top_margin = (height - new_height) // 2
            left_margin = (width - new_width) // 2

            # Memotong area gambar
            face_crop = face_crop[top_margin:top_margin + new_height, left_margin:left_margin + new_width]
            face_crop = cv2.resize(face_crop, (200, 200))
            return face_crop
        return None
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def extract_features_and_predict(image, model):
    """Extract features using Canny edges and predict age."""
    def features_grid(img):
        features = []
        for y in range(0, img.shape[0], 10):
            for x in range(0, img.shape[1], 10):
                section = img[y:y+10, x:x+10]
                features.extend([np.mean(section), np.std(section), np.sum(section)])
        return np.array(features, dtype='float32')

    def extract_canny_edges(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (200, 200))
        
        # Calculate the median of the pixel intensities
        median = np.median(img_resized)

        # Define thresholds based on sigma
        sigma = 0.9  # adjust this as needed
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        # Apply Canny edge detection with dynamic thresholds
        img_canny = cv2.Canny(img_resized, lower, upper)
        return features_grid(img_canny), img_canny

    features, canny_image = extract_canny_edges(image)
    if features is None:
        return "Face not detected!"
    
    features = features.reshape(1, -1)
    predicted_age = model.predict(features)[0]
    return int(predicted_age), features.tolist(), canny_image

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
      <head>
        <title>Home</title>
      </head>
      <body>
        <h1>Home</h1>
        <form action="/age_detection" method="POST" enctype="multipart/form-data">
            <input type="text" name="recognition_path" placeholder="Enter Image URL">
            <input type="submit" value="Detect Age">
        </form>
      </body>
    </html>
    '''
@app.route('/age_detection', methods=['POST'])
def age_detection():
    try:
        # Ambil URL gambar dari form
        recognition_path = request.form.get('recognition_path')
        if not recognition_path:
            return jsonify({"error": "Image URL is required"}), 400
        
        # Unduh gambar
        image = download_image(recognition_path)
        if image is None:
            return jsonify({"error": "Failed to download the image"}), 400
        
        # Deteksi wajah
        cropped_face = face_detection(image)
        if cropped_face is None:
            return jsonify({"error": "Face not detected!"}), 400

        # Prediksi usia
        predicted_age, feature, canny_image = extract_features_and_predict(cropped_face, model)

        # Konversi cropped_face ke base64
        cropped_face_image = Image.fromarray(cropped_face)
        buffered = BytesIO()
        cropped_face_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Konversi canny_image ke PIL dan base64
        canny_image_pil = Image.fromarray(canny_image)
        buffered.seek(0)
        buffered.truncate(0)
        canny_image_pil.save(buffered, format="JPEG")
        image_canny64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Render template dengan data
        return render_template(
            'index.html',
            image=f"data:image/jpeg;base64,{image_base64}",
            age=predicted_age,
            feature=feature,
            canny_img=f"data:image/jpeg;base64,{image_canny64}"
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# @app.route('/age_detection', methods=['POST'])
# def age_detection():
#     try:
#         # Ambil URL gambar dari form
#         recognition_path = request.form.get('recognition_path')
#         if not recognition_path:
#             return jsonify({"error": "Image URL is required"}), 400
        
#         # Unduh gambar
#         image = download_image(recognition_path)
#         if image is None:
#             return jsonify({"error": "Failed to download the image"}), 400
        
#         # Deteksi wajah
#         cropped_face = face_detection(image)
#         if cropped_face is None:
#             return jsonify({"error": "Face not detected!"}), 400

#         # Prediksi usia
#         predicted_age, feature, canny_image = extract_features_and_predict(cropped_face, model)

#         # Konversi cropped_face ke base64
#         cropped_face_image = Image.fromarray(cropped_face)
#         buffered = BytesIO()
#         cropped_face_image.save(buffered, format="JPEG")
#         image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
#         # Convert canny_image (NumPy array) to PIL Image
#         canny_image_pil = Image.fromarray(canny_image)
    
#         # Reset buffered stream untuk penggunaan ulang
#         buffered.seek(0)
#         buffered.truncate(0)
#         canny_image_pil.save(buffered, format="JPEG")
#         image_canny64 = base64.b64encode(buffered.getvalue()).decode("utf-8")


#         # Render template dengan data
#         return render_template(
#             'index.html',
#             image=f"data:image/jpeg;base64,{image_base64}",
#             age=predicted_age,
#             feature=feature,
#             canny_img=f"data:image/jpeg;base64,{image_canny64}"
#         )
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
