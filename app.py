import random
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
model = joblib.load('assets/nn_20_canny_model_0.58.pkl')  # Ganti dengan path model Anda

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
def face_detection(image, size=(200, 200)):
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
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Jika wajah terdeteksi, potong wajah pertama
        if len(faces) > 0:
            # Cari wajah dengan frame terbesar
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Pilih berdasarkan area (w * h)
            x, y, w, h = largest_face
            
            # Memotong gambar sesuai area deteksi wajah terbesar
            face_crop = image[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, size)

            # Mengurangi area gambar sebesar 10% (90% dari ukuran asli)
            height, width = face_crop.shape[:2]
            new_height = int(height * 0.70)
            new_width = int(width * 0.70)

            # Menghitung margin untuk cropping agar tetap di tengah
            top_margin = (height - new_height) // 2
            left_margin = (width - new_width) // 2

            # Memotong area gambar
            face_crop = face_crop[top_margin:top_margin + new_height, left_margin:left_margin + new_width]
            face_crop = cv2.resize(face_crop, size)
            return face_crop

        print("No face detected.")
        return None
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return None
    except Exception as e:
        print(f"General error: {e}")
        return None

# Fungsi untuk mengekstrak fitur dan klasifikasi umur
def extract_features_and_predict(image, model):
    def features_quadrants(img):
        h, w = img.shape  # Ambil tinggi dan lebar gambar
        h_step, w_step = h // 20, w // 20  # Ukuran setiap kuadran (dibagi 20x20)

        features = []
        for i in range(20):  # Iterasi untuk setiap baris kuadran
            for j in range(20):  # Iterasi untuk setiap kolom kuadran
                # Potong kuadran berdasarkan indeks
                quad = img[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
                features.append(np.mean(quad))  # Hitung mean kuadran
                features.append(np.std(quad))   # Hitung std kuadran
                features.append(np.sum(quad))   # Hitung sum kuadran

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
        return features_quadrants(img_canny)

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
            return jsonify({"message": "Face not detected!", "predicted_age": 6})

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
