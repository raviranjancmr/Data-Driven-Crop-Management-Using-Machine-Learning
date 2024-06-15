from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Prepare features for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Transform features
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    # Map prediction to crop and image file
    crop_dict = {
        1: "Rice",
        2: "Maize",
        3: "Jute",
        4: "Cotton",
        5: "Coconut",
        6: "Papaya",
        7: "Orange",
        8: "Apple",
        9: "Muskmelon",
        10: "Watermelon",
        11: "Grapes",
        12: "Mango",
        13: "Banana",
        14: "Pomegranate",
        15: "Lentil",
        16: "Blackgram",
        17: "Mungbean",
        18: "Mothbeans",
        19: "Pigeonpeas",
        20: "Kidneybeans",
        21: "Chickpea",
        22: "Coffee"
    }
    
    # Map crop names to image filenames
    crop_image_mapping = {
        "Rice": "https://images.pexels.com/photos/164504/pexels-photo-164504.jpeg",
        "Maize": "https://images.pexels.com/photos/547264/pexels-photo-547264.jpeg",
        "Jute": "https://images.pexels.com/photos/14251547/pexels-photo-14251547.jpeg",
        "Cotton": "https://images.pexels.com/photos/5474394/pexels-photo-5474394.jpeg",
        "Coconut": "https://images.pexels.com/photos/7676779/pexels-photo-7676779.jpeg",
        "Papaya": "https://images.pexels.com/photos/5217968/pexels-photo-5217968.jpeg",
        "Orange": "https://images.pexels.com/photos/327098/pexels-photo-327098.jpeg",
        "Apple": "https://images.pexels.com/photos/209439/pexels-photo-209439.jpeg",
        "Muskmelon": "https://media.istockphoto.com/id/480915274/photo/fresh-melon.jpg?b=1&s=612x612&w=0&k=20&c=Oh64YXvB9ViROZoO41y9PtPRs4zsp67UmTeJRHpoITY=",
        "Watermelon": "https://images.pexels.com/photos/1313267/pexels-photo-1313267.jpeg",
        "Grapes": "https://images.pexels.com/photos/60021/grapes-wine-fruit-vines-60021.jpeg",
        "Mango": "https://images.pexels.com/photos/2294471/pexels-photo-2294471.jpeg",
        "Pomegranate": "https://images.pexels.com/photos/4869085/pexels-photo-4869085.jpeg",
        "Lentil": "https://images.pexels.com/photos/6086414/pexels-photo-6086414.jpeg",
        "Blackgram": "https://media.istockphoto.com/id/1158693674/photo/dry-organic-murad-split-matpe-beans.jpg?b=1&s=612x612&w=0&k=20&c=cW9B8Pcjqc03EwtzcpkBvL25NqElF0Hu7Rsw6rK4ois=",
        "Mungbean": "https://media.istockphoto.com/id/1310279351/photo/macro-close-up-of-organic-green-gram-or-whole-green-moong-dal-on-a-white-ceramic-soup-spoon.jpg?b=1&s=612x612&w=0&k=20&c=bFzTVvS8JWM-Au9lkj_GFavFkKhYT1NVI1K5l1C04Fo=",
        "Mothbeans": "https://media.istockphoto.com/id/1310610629/photo/full-frame-shot-of-turkish-gram-grains-forming-texture.jpg?b=1&s=612x612&w=0&k=20&c=RcgMen1bfwMkpYT-pwfM9iNuyoliB-fKtUfl-_scqhI=",
        "Pigeonpeas": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7v-o3QG6B4ERJ_8m1pBphaFb13Fc0YmYNHw&s",
        "Kidneybeans": "https://images.pexels.com/photos/6316673/pexels-photo-6316673.jpeg",
        "Chickpea": "https://images.pexels.com/photos/14440733/pexels-photo-14440733.jpeg",
        "Coffee": "https://images.pexels.com/photos/1695052/pexels-photo-1695052.jpeg"
    }

    # Determine the crop and image filename
    crop_id = prediction[0]
    crop = crop_dict.get(crop_id)
    image_filename = crop_image_mapping.get(crop, "default.png")  # Default image if not found

    # Format the result message
    if crop:
        result = f"{crop} is the best crop to be cultivated right there."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Render template and pass result and image filename
    return render_template('index.html', result=result, image_filename=image_filename)

if __name__ == "__main__":
    app.run(debug=True)
