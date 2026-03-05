import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from openai import OpenAI
import base64

# Configure OpenAI client
client = OpenAI(
    api_key=""
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@app.route("/")
def index():
    return render_template("medical_analyzer.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Encode image
        base64_image = encode_image(filepath)
        
        # Analyze with OpenAI Vision API (using agentic approach)
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert medical image analysis AI assistant. Your role is to analyze medical images 
                    (X-rays, CT scans, MRIs, ultrasounds, dermatology images, etc.) and provide:
                    Detailed observations about what you see in the image
                    
                    
                    Be thorough, professional, and always emphasize that this is an AI analysis tool and not a replacement 
                    for professional medical diagnosis. Use medical terminology appropriately but explain findings clearly."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this medical image in detail. Provide comprehensive observations and recommendations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            "analysis": analysis,
            "filename": filename
        })
        
    except Exception as e:
        # Clean up on error
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except:
            pass
        
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
