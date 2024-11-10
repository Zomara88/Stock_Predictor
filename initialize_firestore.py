
from firebase_admin import credentials, firestore
import firebase_admin

# Initialize Firebase Admin SDK
cred = credentials.Certificate('Path-to-firebase-adminsdk-json-file')
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

def initialize_firestore():
    # Create a sample user to ensure the collection is set up correctly
    users_ref = db.collection('users')
    user_doc = users_ref.document('sample_user')
    user_doc.set({
        'username': 'sample_user',
        'email': 'sample_user@example.com',
        'password': 'sample_password'
    })
    print("Firestore collection initialized successfully.")

if __name__ == "__main__":
    initialize_firestore()
