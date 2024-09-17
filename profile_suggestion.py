# Import necessary libraries
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Synthesize user data with unique profile IDs
num_users = 1000
num_profiles_per_user = 3

user_ids = np.arange(num_users)
profile_ids = np.arange(num_users * num_profiles_per_user)
profile_types = np.random.choice(['ACQ', 'ISS'], size=(num_users, num_profiles_per_user))

# Create a dataframe
df = pd.DataFrame({
    'user_id': np.repeat(user_ids, num_profiles_per_user),
    'profile_id': profile_ids,
    'profile_type': profile_types.flatten()
})

# Encode categorical variables
le = LabelEncoder()
df['profile_type_encoded'] = le.fit_transform(df['profile_type'])

# Split data into training and testing sets
train_user_ids, test_user_ids, train_profiles, test_profiles = train_test_split(
    df['user_id'], df['profile_type_encoded'], test_size=0.2, random_state=42)

# Define the model
num_users += 1  # Account for 0-indexing
input_layer = Input(shape=(1,), name='user_input')
x = Embedding(input_dim=num_users, output_dim=128, name='user_embedding')(input_layer)
x = Dense(64, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_user_ids, train_profiles, epochs=10, batch_size=128, validation_data=(test_user_ids, test_profiles))

# Evaluate the model
loss, accuracy = model.evaluate(test_user_ids, test_profiles)
print(f'Test accuracy: {accuracy:.3f}')

# Define a dictionary to store profile suggestions
profile_suggestions = {
    'ACQ': [],
    'ISS': []
}

# Iterate over the unique profile types
for profile_type in df['profile_type'].unique():
    # Filter profiles with the current profile type
    profile_ids = df[df['profile_type'] == profile_type]['profile_id']

    # Get the top-N profiles with the highest embedding similarity
    suggested_profile_ids = profile_ids.iloc[:3]  # Adjust N as needed

    # Store the suggested profile IDs
    profile_suggestions[profile_type] = suggested_profile_ids.tolist()

# Print the profile suggestions
print(profile_suggestions)

# Use the model to predict the profile type for a new user
new_user_id = np.array([42])
predicted_profile = model.predict(new_user_id)
predicted_profile_type = 'ACQ' if np.argmax(predicted_profile) == 0 else 'ISS'
print(f'Predicted profile type for user {new_user_id[0]}: {predicted_profile_type}')

# Suggest profiles based on the predicted type
suggested_profiles = profile_suggestions[predicted_profile_type]
print(f'Suggested profiles for user {new_user_id[0]}: {suggested_profiles}')