# AIModels/train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split

# Generate or load interaction data - replace with actual data as needed
# Example data format: features like response time, interaction accuracy, and previous attempts
# Labels include comprehension level (0 for low, 1 for high) and suggested haptic intensity
# Here, each feature set is [response_time, interaction_accuracy, num_attempts]
features = np.array([
    [1.2, 0.85, 2],
    [0.8, 0.95, 1],
    [1.5, 0.60, 3],
    [0.6, 0.90, 1],
    [1.0, 0.70, 2],
    [1.3, 0.65, 2]
])

# Label structure: comprehension (0 = needs help, 1 = proficient) and haptic intensity level (0-1)
labels = np.array([
    [0, 0.3],   # Comprehension low, low haptic intensity
    [1, 0.7],   # Comprehension high, higher haptic intensity
    [0, 0.4],
    [1, 0.8],
    [0, 0.5],
    [0, 0.6]
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model to predict comprehension level and haptic intensity
def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='linear')  # Output: [comprehension, haptic intensity]
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Initialize model
model = create_model(input_shape=X_train.shape[1])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Model test loss: {loss}, Mean Absolute Error: {mae}")

# Save the trained model for Unity integration
model.save('AIModels/adaptive_hologram_model.h5')
print("Model saved as adaptive_hologram_model.h5")

