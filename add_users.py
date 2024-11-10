import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
import hashlib

# Load environment variables from .env file
load_dotenv()

# Initialize Firebase Admin SDK
firebase_cert_path = os.getenv('FIREBASE_CERT_PATH')
cred = credentials.Certificate(firebase_cert_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

def add_user(username, email, password):
    # Hash the password before storing it
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Store user data in Firestore
    users_ref = db.collection('users')
    user_doc = users_ref.document(username)
    user_doc.set({
        'username': username,
        'email': email,
        'password': hashed_password
    })
    print(f"User {username} added successfully.")

if __name__ == "__main__":
    add_user('example_user', 'example_user@example.com', 'password123')
    # Add more users as needed