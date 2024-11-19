from flask import Flask, send_from_directory
from api import api_bp
from flask_cors import CORS
import json

with open('config.json') as f:
    config = json.load(f)

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["config"] = config
# Serve Angular app as the index route
@app.route("/")
def index():
    return send_from_directory("dist/daphne-ui", "index.html")

# Serve static files from the Angular app folder
@app.route("/<path:filename>")
def static_files(filename):
    # Exclude routes with dots (to avoid serving static files)
    if "." not in filename:
        return send_from_directory("dist/daphne-ui", "index.html")
    else:
        return send_from_directory("dist/daphne-ui", filename)


# Register the API blueprint
app.register_blueprint(api_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
