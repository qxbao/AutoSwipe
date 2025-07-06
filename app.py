from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import Database
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "The server is running just fine!"

@app.route('/record', methods=['POST'])
def record_swipe():
    try:
        body = request.json
        images = body.get('images', [])
        age = body.get('age')
        score = body.get('score', 0.0)
        if not images or age is None:
            return jsonify({"status": "Error", "message": "No images or age provided"}), 400
        profile_id = str(uuid.uuid4())
        profile_folder = Database.save_profile_folder(profile_id, images)
        db = Database()
        db.save_profile(profile_id, age, len(images), profile_folder, score)
        db.close()
        return jsonify({
            "status": "Success",
            "message": "Profile recorded successfully",
            "profile_id": profile_id,
            "score": score,
            "num_images": len(images)
        }), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

if __name__ == '__main__':
    Database.init_db()
    app.run(debug=True, port=5000)