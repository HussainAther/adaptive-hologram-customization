# AIModels/haptic_adaptive_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_haptic_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Output: haptic intensity level
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example training data: interaction features, haptic intensity levels
features = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Replace with actual data
haptic_intensity = [0.5, 0.8]  # Intensity levels based on student need

# Train model
model = create_haptic_model(input_shape=len(features[0]))
model.fit(features, haptic_intensity, epochs=10, batch_size=2)
model.save('AIModels/haptic_adaptive_model.h5')

