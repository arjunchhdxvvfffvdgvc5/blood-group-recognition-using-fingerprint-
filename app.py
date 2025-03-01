from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern
import os
import joblib
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from glob import glob
import time
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Add after app initialization
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

# Mock user database (replace with real database in production)
users = {
    'admin': generate_password_hash('admin123'),
    'user1': generate_password_hash('password123'),  # Add new user
    'testuser': generate_password_hash('test123')    # Add another user
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class FingerprintProcessor:
    @staticmethod
    def enhance_fingerprint(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    @staticmethod
    def extract_features(image):
        resized = cv2.resize(image, (200, 200))
        features = []

        # 1. LBP Features (Fixed bin count to 258)
        lbp = FingerprintProcessor.get_lbp_features(resized)
        features.extend(lbp)

        # 2. Ridge Orientation Features
        orientations = FingerprintProcessor.get_ridge_orientations(resized)
        features.extend(orientations)

        # 3. Ridge Density Features
        density = FingerprintProcessor.get_ridge_density(resized)
        features.extend(density)

        # Ensure feature count is correct
        print(f"Debug: Extracted {len(features)} features (Expected: 292)")
        return np.array(features)

    @staticmethod
    def get_lbp_features(image):
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=258, range=(0, 258))  # Fixed bins to 258
        return hist

    @staticmethod
    def get_ridge_orientations(image):
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        hist, _ = np.histogram(orientation, bins=18, range=(0, 180))
        return hist

    @staticmethod
    def get_ridge_density(image):
        blocks = [image[i:i+50, j:j+50] for i in range(0, 200, 50) for j in range(0, 200, 50)]
        densities = [np.sum(block > 127) / (50 * 50) for block in blocks]
        return densities

class BloodGroupClassifier:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join('models', 'blood_group_model.pkl')
        self.load_model()  # Try to load existing model
        
        # If no model exists, create a new one
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
    def load_model(self):
        """Load the trained model if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def train_model(self, X_train, y_train):
        """Train the model with new data"""
        # Ensure model exists
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Save the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
    def predict(self, features):
        """Make prediction using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train the model first.")
            
        prediction = self.model.predict([features])[0]
        confidence = np.max(self.model.predict_proba([features]))
        return prediction, confidence

def process_fingerprint(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    processor = FingerprintProcessor()
    enhanced = processor.enhance_fingerprint(image)
    features = processor.extract_features(enhanced)  

    classifier = BloodGroupClassifier()
    prediction, confidence = classifier.predict(features)

    return prediction, confidence, enhanced

# Add password validation
def is_valid_password(password):
    """Check if password meets security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        login_type = request.form.get('login_type', 'user')

        if not username or not password:
            return render_template('login.html', error='Please fill in all fields')

        if username not in users:
            return render_template('login.html', error='Invalid username or password')

        if not check_password_hash(users[username], password):
            return render_template('login.html', error='Invalid username or password')

        # Check if trying to access admin with non-admin account
        if login_type == 'admin' and username != 'admin':
            return render_template('login.html', error='Invalid admin credentials')

        session['username'] = username
        
        # Redirect admin to dashboard
        if username == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

import mediapipe as mp

def is_fingerprint_image(image_path):
    """Checks if the uploaded image is a fingerprint (not a face or random image)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, "Invalid image format. Please upload a clear fingerprint image."

    # Check if image contains a face (to reject faces)
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    image_rgb = cv2.imread(image_path)  # Read in RGB format
    results = mp_face_detection.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

    if results.detections:  # If face is detected
        return False, "Face detected! Please upload a fingerprint image."

    # Simple fingerprint validation: Check texture patterns
    if np.mean(image) < 50:  # If the image is too dark
        return False, "The uploaded image does not appear to be a fingerprint. Please try again."

    return True, "Valid fingerprint image."

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        return jsonify({'error': 'Please login first'}), 401
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded. Please upload a fingerprint image.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected. Please choose a fingerprint image.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG fingerprint image.'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate fingerprint image
        is_valid, message = is_fingerprint_image(filepath)
        if not is_valid:
            os.remove(filepath)  # Remove invalid image
            return jsonify({'error': message}), 400

        # Initialize classifier
        classifier = BloodGroupClassifier()
        
        # Check if model exists
        if classifier.model is None or not hasattr(classifier.model, 'predict'):
            return jsonify({
                'error': 'Model not trained. Please train the model first.'
            }), 503

        # Process fingerprint and get prediction
        blood_group, confidence, enhanced_image = process_fingerprint(filepath)

        # Save enhanced image to static folder for display
        enhanced_filename = 'enhanced_' + filename
        enhanced_path = os.path.join('static', enhanced_filename)
        cv2.imwrite(enhanced_path, enhanced_image)

        # Clean up original image
        os.remove(filepath)

        return render_template('result.html',
                               blood_group=blood_group,
                               confidence=confidence,
                               enhanced_image=enhanced_filename)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Basic validation
        if not username or not password or not confirm_password:
            return render_template('register.html', error='Please fill in all fields')
            
        if username in users:
            return render_template('register.html', error='Username already exists')
            
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
            
        # Add new user with hashed password
        users[username] = generate_password_hash(password)
        
        # Automatically log in the new user
        session['username'] = username
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if 'username' not in session or session['username'] != 'admin':
        return jsonify({'error': 'Only admin can train the model'}), 403
        
    try:
        # Path to your training data
        data_path = 'training_data'
        blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        
        features = []
        labels = []
        
        # Process each blood group folder
        for blood_group in blood_groups:
            group_path = os.path.join(data_path, blood_group)
            if not os.path.exists(group_path):
                continue
                
            # Process all images in the blood group folder
            for img_path in glob(os.path.join(group_path, '*.*')):
                try:
                    # Read and process image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    # Process fingerprint
                    processor = FingerprintProcessor()
                    enhanced = processor.enhance_fingerprint(image)
                    image_features = processor.extract_features(enhanced)
                    
                    features.append(image_features)
                    labels.append(blood_group)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
        
        if not features:
            return jsonify({'error': 'No training data found'}), 400
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        classifier = BloodGroupClassifier()
        classifier.train_model(X_train, y_train)
        
        # Test accuracy
        accuracy = classifier.model.score(X_test, y_test)
        
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': float(accuracy),
            'samples': len(y)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-interface')
def train_interface():
    if 'username' not in session or session['username'] != 'admin':
        return redirect(url_for('login'))
    return render_template('train.html')

@app.route('/contribute', methods=['GET'])
def contribute():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('contribute.html')

@app.route('/contribute-dataset', methods=['POST'])
def contribute_dataset():
    if 'username' not in session:
        return jsonify({'error': 'Please login first'}), 401
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']
        blood_group = request.form.get('blood_group')
        
        if not blood_group:
            return jsonify({'error': 'Please select blood group'}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Create directory for blood group if it doesn't exist
        group_dir = os.path.join('training_data', blood_group)
        os.makedirs(group_dir, exist_ok=True)

        # Save file with unique name
        filename = f"{session['username']}_{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(group_dir, filename)
        file.save(filepath)

        # Return success page with details
        return render_template('contribute_success.html',
                             blood_group=blood_group,
                             date=time.strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'username' not in session or session['username'] != 'admin':
        return redirect(url_for('login'))
        
    # Collect statistics
    stats = {
        'total_users': len(users),
        'total_predictions': get_total_predictions(),
        'dataset_size': get_dataset_size(),
        'model_accuracy': get_model_accuracy()
    }
    
    # Get recent activity
    recent_activity = get_recent_activity()
    
    # Get user list
    user_list = get_user_list()
    
    return render_template('admin_dashboard.html',
                         stats=stats,
                         recent_activity=recent_activity,
                         users=user_list)

@app.route('/admin/reset-password', methods=['POST'])
def reset_password():
    if 'username' not in session or session['username'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.get_json()
    username = data.get('username')
    
    if username not in users:
        return jsonify({'error': 'User not found'}), 404
        
    # Reset password to default
    new_password = 'password123'
    users[username] = generate_password_hash(new_password)
    
    return jsonify({
        'message': f'Password reset for {username}. New password: {new_password}'
    })

@app.route('/admin/delete-user', methods=['POST'])
def delete_user():
    if 'username' not in session or session['username'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.get_json()
    username = data.get('username')
    
    if username not in users or username == 'admin':
        return jsonify({'error': 'Cannot delete this user'}), 400
        
    del users[username]
    return jsonify({'message': f'User {username} deleted successfully'})

# Helper functions
def get_total_predictions():
    # Implement prediction counting logic
    return 100  # Placeholder

def get_dataset_size():
    # Count files in training_data
    count = 0
    for blood_group in ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']:
        path = os.path.join('training_data', blood_group)
        if os.path.exists(path):
            count += len(os.listdir(path))
    return count

def get_model_accuracy():
    # Get current model accuracy
    try:
        classifier = BloodGroupClassifier()
        return 0.85  # Placeholder - implement actual accuracy check
    except:
        return 0

def get_recent_activity():
    # Implement activity logging and retrieval
    return [
        {
            'username': 'user1',
            'action': 'Prediction',
            'blood_group': 'A+',
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': 0.95
        }
        # Add more activities
    ]

def get_user_list():
    # Return list of users with details
    return [
        {
            'username': username,
            'registration_date': datetime.now().strftime("%Y-%m-%d"),
            'contributions': 5  # Placeholder
        }
        for username in users
    ]

if __name__ == '__main__':
    app.run(debug=True)
