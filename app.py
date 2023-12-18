from flask import Flask, render_template, request
from fastai.vision.learner import load_learner
from fastai.vision.core import PILImage
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)

# Load the exported learner
path = Path()
learn_inf = load_learner(path/'export.pkl')

# Dictionary to map model predictions to descriptions and risk factors for plant diseases
# Dictionary to map model predictions to descriptions and risk factors for plant diseases
class_labels = {
    'Apple___Apple_scab': {
        'label': 'Apple Scab',
        'description': 'Apple Scab is a common fungal disease that affects apple trees. It appears as olive-green to black spots on the leaves, often with a scaly texture.',
        'risk_factors': 'High humidity and wet conditions favor the development of Apple Scab.'
    },
    'Apple___Black_rot': {
        'label': 'Apple Black Rot',
        'description': 'Apple Black Rot is a fungal disease that affects apples, causing circular lesions with dark borders on the fruit. It can lead to significant fruit loss.',
        'risk_factors': 'Warm and wet conditions, especially during the growing season, promote the spread of Apple Black Rot.'
    },
    'Apple___Cedar_apple_rust': {
        'label': 'Apple Cedar Apple Rust',
        'description': 'Cedar Apple Rust is a fungal disease affecting apple trees. It causes bright orange spots on leaves and can lead to defoliation in severe cases.',
        'risk_factors': 'Requires both apple and cedar trees for its life cycle. Wet conditions and specific temperature ranges contribute to its spread.'
    },
    'Apple___healthy': {
        'label': 'Healthy Apple',
        'description': 'This category represents a healthy apple plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy apple plants.'
    },
    'Blueberry___healthy': {
        'label': 'Healthy Blueberry',
        'description': 'This category represents a healthy blueberry plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy blueberry plants.'
    },
    'Cherry_(including_sour)___healthy': {
        'label': 'Healthy Cherry',
        'description': 'This category represents a healthy cherry plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy cherry plants.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'label': 'Cherry Powdery Mildew',
        'description': 'Powdery Mildew is a fungal disease that affects cherry trees. It appears as a white powdery substance on the leaves and can lead to reduced fruit quality.',
        'risk_factors': 'Humid conditions and moderate temperatures favor the development of Powdery Mildew.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'label': 'Corn Cercospora Leaf Spot and Gray Leaf Spot',
        'description': 'Cercospora Leaf Spot and Gray Leaf Spot are fungal diseases that affect corn (maize) plants. They manifest as dark spots on the leaves and can lead to reduced yield.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Corn_(maize)___Common_rust_': {
        'label': 'Corn Common Rust',
        'description': 'Common Rust is a fungal disease that affects corn (maize) plants. It appears as small reddish-brown pustules on the leaves and can reduce crop yield.',
        'risk_factors': 'Warm and humid conditions promote the spread of Common Rust.'
    },
    'Corn_(maize)___healthy': {
        'label': 'Healthy Corn',
        'description': 'This category represents a healthy corn (maize) plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy corn plants.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'label': 'Corn Northern Leaf Blight',
        'description': 'Northern Leaf Blight is a fungal disease that affects corn (maize) plants. It appears as cigar-shaped lesions on the leaves and can lead to significant yield loss.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Grape___Black_rot': {
        'label': 'Grape Black Rot',
        'description': 'Black Rot is a fungal disease that affects grapevines. It causes circular lesions with dark borders on the leaves and can lead to reduced grape quality.',
        'risk_factors': 'Warm and wet conditions, especially during the growing season, promote the spread of Grape Black Rot.'
    },
    'Grape___Esca_(Black_Measles)': {
        'label': 'Grape Esca (Black Measles)',
        'description': 'Esca, also known as Black Measles, is a fungal disease that affects grapevines. It can cause wilting, yellowing, and black streaks in the wood.',
        'risk_factors': 'The exact causes are complex, involving various fungi and environmental factors.'
    },
    'Grape___healthy': {
        'label': 'Healthy Grape',
        'description': 'This category represents a healthy grapevine without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy grapevines.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'label': 'Grape Leaf Blight (Isariopsis Leaf Spot)',
        'description': 'Leaf Blight, specifically Isariopsis Leaf Spot, is a fungal disease that affects grapevines. It appears as small spots on the leaves and can lead to defoliation.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'label': 'Citrus Greening (Huanglongbing)',
        'description': 'Citrus Greening, also known as Huanglongbing, is a bacterial disease that affects citrus trees. It can cause yellowing of leaves, stunted growth, and bitter fruit.',
        'risk_factors': 'Spread by a bacterial pathogen and the Asian citrus psyllid insect.'
    },
    'Peach___Bacterial_spot': {
        'label': 'Peach Bacterial Spot',
        'description': 'Bacterial Spot is a disease that affects peach trees. It appears as small, dark lesions on the leaves and fruit, leading to reduced peach quality.',
        'risk_factors': 'Warm and humid conditions favor the development of Peach Bacterial Spot.'
    },
    'Peach___healthy': {
        'label': 'Healthy Peach',
        'description': 'This category represents a healthy peach tree without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy peach trees.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'label': 'Bell Pepper Bacterial Spot',
        'description': 'Bacterial Spot is a disease that affects bell pepper plants. It appears as small, dark lesions on the leaves and fruit, leading to reduced pepper quality.',
        'risk_factors': 'Warm and humid conditions favor the development of Bell Pepper Bacterial Spot.'
    },
    'Pepper,_bell___healthy': {
        'label': 'Healthy Bell Pepper',
        'description': 'This category represents a healthy bell pepper plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy bell pepper plants.'
    },
        'Potato___Early_blight': {
        'label': 'Potato Early Blight',
        'description': 'Early Blight is a fungal disease that affects potato plants. It appears as dark lesions with concentric rings on the leaves and can lead to reduced potato yield.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Potato___healthy': {
        'label': 'Healthy Potato',
        'description': 'This category represents a healthy potato plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy potato plants.'
    },
    'Potato___Late_blight': {
        'label': 'Potato Late Blight',
        'description': 'Late Blight is a fungal disease that affects potato plants. It appears as water-soaked lesions on the leaves and can lead to significant crop loss.',
        'risk_factors': 'Favorable conditions include cool and wet weather.'
    },
    'Raspberry___healthy': {
        'label': 'Healthy Raspberry',
        'description': 'This category represents a healthy raspberry plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy raspberry plants.'
    },
    'Soybean___healthy': {
        'label': 'Healthy Soybean',
        'description': 'This category represents healthy soybean plants without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy soybean plants.'
    },
    'Squash___Powdery_mildew': {
        'label': 'Squash Powdery Mildew',
        'description': 'Powdery Mildew is a fungal disease that affects squash plants. It appears as white, powdery spots on the leaves and can lead to reduced fruit quality.',
        'risk_factors': 'Warm and dry conditions favor the development of Powdery Mildew.'
    },
    'Strawberry___healthy': {
        'label': 'Healthy Strawberry',
        'description': 'This category represents a healthy strawberry plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy strawberry plants.'
    },
    'Strawberry___Leaf_scorch': {
        'label': 'Strawberry Leaf Scorch',
        'description': 'Leaf Scorch is a disease that affects strawberry plants. It appears as brown, scorched areas on the leaves and can lead to reduced fruit quality.',
        'risk_factors': 'Favorable conditions include warm and dry weather.'
    },
    'Tomato___Bacterial_spot': {
        'label': 'Tomato Bacterial Spot',
        'description': 'Bacterial Spot is a disease that affects tomato plants. It appears as small, dark lesions on the leaves and fruit, leading to reduced tomato quality.',
        'risk_factors': 'Warm and humid conditions favor the development of Tomato Bacterial Spot.'
    },
    'Tomato___Early_blight': {
        'label': 'Tomato Early Blight',
        'description': 'Early Blight is a fungal disease that affects tomato plants. It appears as dark lesions with concentric rings on the leaves and can lead to reduced tomato yield.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Tomato___healthy': {
        'label': 'Healthy Tomato',
        'description': 'This category represents a healthy tomato plant without any detectable diseases or abnormalities.',
        'risk_factors': 'No specific risk factors for healthy tomato plants.'
    },
    'Tomato___Late_blight': {
        'label': 'Tomato Late Blight',
        'description': 'Late Blight is a fungal disease that affects tomato plants. It appears as water-soaked lesions on the leaves and can lead to significant crop loss.',
        'risk_factors': 'Favorable conditions include cool and wet weather.'
    },
    'Tomato___Leaf_Mold': {
        'label': 'Tomato Leaf Mold',
        'description': 'Leaf Mold is a fungal disease that affects tomato plants. It appears as yellow, moldy spots on the leaves and can lead to reduced fruit quality.',
        'risk_factors': 'Favorable conditions include high humidity and poor air circulation.'
    },
    'Tomato___Septoria_leaf_spot': {
        'label': 'Tomato Septoria Leaf Spot',
        'description': 'Septoria Leaf Spot is a fungal disease that affects tomato plants. It appears as small dark spots with light centers on the leaves and can lead to defoliation.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'label': 'Tomato Spider Mites (Two-spotted Spider Mite)',
        'description': 'Spider Mites are pests that affect tomato plants. They suck plant juices, causing stippling, discoloration, and reduced plant vigor.',
        'risk_factors': 'Hot and dry conditions favor the development of Spider Mites.'
    },
    'Tomato___Target_Spot': {
        'label': 'Tomato Target Spot',
        'description': 'Target Spot is a fungal disease that affects tomato plants. It appears as concentric rings with a target-like appearance on the leaves and can lead to defoliation.',
        'risk_factors': 'Favorable conditions include warm and humid weather.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'label': 'Tomato Mosaic Virus',
        'description': 'Tomato Mosaic Virus is a viral disease that affects tomato plants. It causes mosaic-like patterns on the leaves, stunted growth, and reduced fruit quality.',
        'risk_factors': 'Spread by infected plant material, contaminated tools, and sap-feeding insects.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'label': 'Tomato Yellow Leaf Curl Virus',
        'description': 'Yellow Leaf Curl Virus is a viral disease that affects tomato plants. It causes yellowing, curling, and stunting of leaves, leading to reduced fruit production.',
        'risk_factors': 'Transmitted by whiteflies and can spread rapidly in warm climates.'
    },
}



def predict_plant_disease(img_path):
    img = PILImage.create(img_path)
    predicted_label, _, _ = learn_inf.predict(img)
    return predicted_label

# routes
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/try_now.html", methods=['GET', 'POST'])
def try_now():
    return render_template("try_now.html")

@app.route("/analyse.html", methods=['GET', 'POST'])
def main():
    return render_template("analyse.html")



@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + secure_filename(img.filename)

        # Ensure the 'static' folder exists
        static_folder = Path("static")
        static_folder.mkdir(exist_ok=True)

        img.save(img_path)
        prediction = predict_plant_disease(img_path)

        # Map the model prediction to the description and risk factors
        result = class_labels.get(prediction, {'label': 'Unknown', 'description': '', 'risk_factors': ''})

    return render_template("analyse.html", prediction=result['label'], description=result['description'], risk_factors=result['risk_factors'], img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
