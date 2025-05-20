import pickle
from collections import defaultdict
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from features import get_deep_speaker_embedding # Changed import

class ModelInterface:

    def __init__(self):
        self.features = defaultdict(list) # Stores list of embeddings for each speaker during enrollment
        self.speaker_profiles = {}      # Stores a single representative embedding per speaker after training

    def enroll(self, name, fs, signal):
        """
        Enrolls a speaker by extracting deep speaker embedding and storing it.
        """
        print(f"Enrolling {name}...")
        embedding = get_deep_speaker_embedding(fs, signal)
        if embedding is not None:
            self.features[name].append(embedding)
            print(f"Successfully extracted and stored embedding for {name}.")
        else:
            print(f"Failed to extract embedding for {name}. Signal might be too short or invalid.")

    def train(self):
        """
        Trains the model by computing an average embedding for each enrolled speaker.
        """
        print("Starting training...")
        start_time = time.time()
        self.speaker_profiles = {} # Reset speaker profiles
        
        num_enrolled_speakers = len(self.features)
        if num_enrolled_speakers == 0:
            print("No speakers enrolled. Training cannot proceed.")
            return

        for name, embeddings_list in self.features.items():
            if embeddings_list:
                # Convert list of embeddings to a NumPy array
                embeddings_array = np.array(embeddings_list)
                # Calculate the average embedding across all enrollment samples for the speaker
                mean_embedding = np.mean(embeddings_array, axis=0)
                self.speaker_profiles[name] = mean_embedding
                print(f"Created profile for {name} with {len(embeddings_list)} embeddings.")
            else:
                print(f"No features found for speaker {name}, skipping profile creation.")
        
        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.2f} seconds. {len(self.speaker_profiles)} speaker profiles created.")

    def dump(self, fname):
        """
        Dumps the model (speaker features and profiles) to a file using pickle.
        """
        print(f"Dumping model to {fname}...")
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL) # Use a specific protocol
            print("Model dumped successfully.")
        except Exception as e:
            print(f"Error dumping model: {e}")

    def predict(self, fs, signal):
        """
        Predicts the speaker of a given audio signal using cosine similarity against speaker profiles.
        Returns the predicted speaker name and the similarity score.
        """
        #print("Starting prediction...")
        test_embedding = get_deep_speaker_embedding(fs, signal)

        if test_embedding is None:
            print("Failed to extract embedding for prediction. Cannot predict.")
            return "Unknown", 0.0

        if not self.speaker_profiles:
            print("No speaker profiles available for prediction. Please train the model first.")
            return "Unknown", 0.0

        best_speaker = "Unknown"
        best_score = -1.0  # Cosine similarity ranges from -1 to 1

        # Reshape test_embedding for cosine_similarity: (1, embedding_dim)
        test_embedding_reshaped = test_embedding.reshape(1, -1)

        for name, profile_embedding in self.speaker_profiles.items():
            # Reshape profile_embedding for cosine_similarity: (1, embedding_dim)
            profile_embedding_reshaped = profile_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            score = cosine_similarity(test_embedding_reshaped, profile_embedding_reshaped)[0][0]
            #print(f"Similarity with {name}: {score:.4f}") # Debugging line

            if score > best_score:
                best_score = score
                best_speaker = name
        
        #print(f"Prediction result: Speaker = {best_speaker}, Score = {best_score:.4f}")
        return best_speaker, best_score

    @staticmethod
    def load(fname):
        """
        Loads a model from a dumped pickle file.
        """
        print(f"Loading model from {fname}...")
        try:
            with open(fname, 'rb') as f:
                model_instance = pickle.load(f)
            # Basic validation if the loaded object has the expected new attribute
            if not hasattr(model_instance, 'speaker_profiles'):
                print("Warning: Loaded model might be from an older version (missing 'speaker_profiles').")
                # Optionally, initialize speaker_profiles if it's missing for backward compatibility
                # model_instance.speaker_profiles = {} 
            print("Model loaded successfully.")
            return model_instance
        except FileNotFoundError:
            print(f"Error: Model file '{fname}' not found.")
            return None # Or raise error
        except Exception as e:
            print(f"Error loading model: {e}")
            return None # Or raise error
