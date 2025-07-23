from flask import Flask
from routes import upload

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here'  # Replace with a secure random key

# Register the blueprint only if not already registered
if 'upload' not in [bp.name for bp in app.blueprints.values()]:
    app.register_blueprint(upload)

if __name__ == '__main__':
    app.run(debug=True, port=5001)