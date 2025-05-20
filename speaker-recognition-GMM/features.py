import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from python_speech_features import mfcc
import numpy as np
import torch

# Global variable to hold the loaded SpeechBrain model
SPEAKER_EMBEDDING_MODEL_CLASSIFIER = None

def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, fs)
    if len(mfcc_feature) == 0:
        # Changed for Python 3 compatibility and clearer error messaging
        print(f"ERROR.. failed to extract mfcc feature for a signal of length: {len(signal)}")
    return mfcc_feature

def get_deep_speaker_embedding(fs, signal):
    """
    Extracts speaker embeddings using a pre-trained SpeechBrain model.

    Args:
        fs (int): Sampling rate of the audio signal.
        signal (np.ndarray): Audio signal as a NumPy array.

    Returns:
        np.ndarray: Speaker embedding as a NumPy array, or None if an error occurs.
    """
    global SPEAKER_EMBEDDING_MODEL_CLASSIFIER

    try:
        # Load model if not already loaded
        if SPEAKER_EMBEDDING_MODEL_CLASSIFIER is None:
            print("Loading SpeechBrain speaker embedding model (ECAPA-TDNN)...")
            # Using run_opts to ensure model is loaded on CPU
            SPEAKER_EMBEDDING_MODEL_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"} 
            )
            print("SpeechBrain model loaded successfully on CPU.")

        # Signal Preparation
        # Convert NumPy signal to PyTorch tensor
        audio_tensor = torch.from_numpy(signal.astype(np.float32))

        # Ensure the signal is mono.
        # SpeechBrain models typically expect mono audio.
        # If audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1: # Check for multiple channels e.g. [channels, samples]
        #     print(f"Signal has multiple channels ({audio_tensor.shape[0]}), averaging to mono.")
        #     audio_tensor = torch.mean(audio_tensor, dim=0) 
        # elif audio_tensor.ndim > 1 and audio_tensor.shape[0] == 1: # Case like [1, N]
        #      audio_tensor = audio_tensor.squeeze(0) # Make it [N] for consistency
        
        # The above explicit mono conversion might be too simple.
        # `load_audio` from SpeechBrain handles this robustly.
        # However, we are passed a numpy array.
        # Let's assume the input `signal` is already mono, or handle it if necessary.
        # For now, we'll proceed assuming it's mono or SpeechBrain handles it.
        # Based on "Make sure your input tensor is compliant with the expected sampling rate if you use encode_batch"
        # it's safer to ensure mono and correct sample rate before encode_batch.

        if audio_tensor.ndim > 1: # If it's not 1D
            if audio_tensor.shape[0] > 1 and audio_tensor.shape[1] > 1 : # Stereo [C, N]
                 print(f"Signal appears to be stereo with shape {audio_tensor.shape}, averaging to mono.")
                 audio_tensor = torch.mean(audio_tensor, dim=0)
            elif audio_tensor.shape[0] == 1 : # Mono but in shape [1,N]
                 audio_tensor = audio_tensor.squeeze(0) # Change to [N]
            # Add other cases if necessary, e.g. if samples are in dim 0

        # Resample if necessary - ECAPA-TDNN model expects 16kHz
        model_expected_sr = 16000 
        if fs != model_expected_sr:
            #print(f"Resampling signal from {fs} Hz to {model_expected_sr} Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=model_expected_sr)
            audio_tensor = resampler(audio_tensor) # resampler expects [..., time]
        
        # Reshape to [batch_size, num_samples] as expected by encode_batch
        # At this point, audio_tensor should be 1D [num_samples]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # [1, num_samples]
        elif audio_tensor.ndim == 2 and audio_tensor.shape[0] != 1:
            # This case should ideally not happen if mono conversion was correct
            print(f"Warning: audio_tensor has shape {audio_tensor.shape} before encode_batch, expected [1, num_samples].")


        # Ensure tensor is on the CPU, as the model was loaded on CPU.
        audio_tensor = audio_tensor.to("cpu")

        # Embedding Extraction
        # The model's encode_batch expects a batch of waveforms.
        embeddings = SPEAKER_EMBEDDING_MODEL_CLASSIFIER.encode_batch(audio_tensor)
        
        # Squeeze to get a 1D embedding vector: 
        # Output of encode_batch is typically [batch_size, num_channels_out, embedding_dim]
        # For ECAPA-TDNN, it's [batch_size, 1, 192] (embedding_dim = 192)
        # Squeeze out the channel dimension (dim 1) and batch dimension (dim 0)
        embedding_numpy = embeddings.squeeze(1).squeeze(0).cpu().numpy()

        return embedding_numpy

    except Exception as e:
        print(f"Error extracting deep speaker embedding: {e}")
        # You might want to log the error more formally or raise it depending on desired behavior
        return None
