# create_test_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create sample training data
X = np.random.rand(1000, 292)  # 292 features
y = np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], 1000)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/blood_group_model.pkl')
# create_test_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create sample training data
X = np.random.rand(1000, 292)  # 292 features
y = np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], 1000)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/blood_group_model.pkl')
