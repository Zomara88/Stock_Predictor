from firebase_admin import credentials, firestore
import firebase_admin
import hashlib

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/bryan/OneDrive/Desktop/HACKFIN/stocksusers-488b4-firebase-adminsdk-srmmy-10a4a1808c.json')
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