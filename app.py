from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import joblib
from firebase_admin import credentials, initialize_app, firestore
import os

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('config/firebase-credentials.json')  # Ganti dengan path kredensial Firebase Anda
initialize_app(cred, {'storageBucket': 'soundfit-bfedd.appspot.com'})  # Ganti dengan ID aplikasi Anda

app = Flask(__name__)

# Load model
model = joblib.load('assets/svc_canny_model_acc_0.463.pkl')  # Ganti dengan path model Anda

# Fungsi untuk mengunduh gambar dari URL Firebase Storage
def download_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Fungsi untuk mendeteksi wajah
def face_detection(image):
    try:
        cascade_path = "assets/haarcascade_frontalface_default.xml"  # Path ke model cascade
        if not os.path.exists(cascade_path):
            raise Exception("Haarcascade file not found at the specified path.")
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Periksa apakah cascade berhasil dimuat
        if face_cascade.empty():
            print("Error: Unable to load the cascade classifier.")
        else:
            print("Cascade classifier loaded successfully.")
        
        # Konversi gambar ke grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        # faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

        
        # Jika wajah terdeteksi, potong wajah pertama
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Ambil koordinat wajah pertama
            return image[y:y+h, x:x+w]  # Crop wajah
        return None
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return None
    except Exception as e:
        print(f"General error: {e}")
        return None

# Fungsi untuk mengekstrak fitur dan klasifikasi umur
# Fungsi untuk mengekstrak fitur dan klasifikasi umur
def extract_features_and_predict(image, model):
    def features_grid(img):
        features = np.array([], dtype='uint8')
        section = 1
        for y in range(0, img.shape[0], 10):
            for x in range(0, img.shape[1], 10):
                section_img = img[y:y+10, x:x+10]
                section_mean = np.mean(section_img)
                section_std = np.std(section_img)
                features = np.append(features, [section_mean, section_std])
        return features

    def extract_canny_edges(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (200, 200))
        img_canny = cv2.Canny(img_resized, threshold1=100, threshold2=200)
        return features_grid(img_canny)

    # Ekstrak fitur dan lakukan prediksi
    features = extract_canny_edges(image)
    if features is None:
        return "Face not detected!"
    
    features = features.reshape(1, -1)
    predicted_age = model.predict(features)[0]
    
    # Konversi hasil prediksi ke tipe data yang bisa diserialisasi JSON
    predicted_age = int(predicted_age)  # Pastikan hasil prediksi adalah integer biasa
    
    return predicted_age


# Route untuk mendeteksi umur berdasarkan gambar
@app.route('/age_detection', methods=['POST'])
def age_detection():
    try:
        recognition_path = request.form.get('recognition_path', type=str)
        if not recognition_path:
            return jsonify({"error": "recognition_path is required"}), 400
        
        print(f"Received recognition_path: {recognition_path}")  # Log untuk memeriksa URL yang diterima
        
        # Mengunduh gambar menggunakan recognition_path
        image = download_image(recognition_path)
        if image is None:
            return jsonify({"error": "Failed to download the image."}), 500

        # Log bentuk gambar untuk debug
        print(f"Image shape: {image.shape}")

        cropped_face = face_detection(image)
        if cropped_face is None:
            return jsonify({"error": "Face not detected!"}), 400

        predicted_age = extract_features_and_predict(cropped_face, model)
        if predicted_age == "Face not detected!":
            return jsonify({"error": predicted_age}), 400
        
        print(f"Predicted Age: {predicted_age}")  # Log prediksi usia

        predicted_age = int(predicted_age)  # Konversi int64 menjadi int biasa
        return jsonify({"message": "Age detection successful", "predicted_age": predicted_age})


    except Exception as e:
        print(f"Error: {str(e)}")  # Log error untuk detail kesalahan
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
