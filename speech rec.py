import pyaudio
import speech_recognition as sr
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import threading

#  (dummy implementation)
def analyze_sentiment(text):
   
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([text])
    model = LogisticRegression()
    model.fit(X, [1])  # Dummy fit, replace with a trained model
    sentiment = model.predict(X)
    return "Positive" if sentiment[0] == 1 else "Negative"


def analyze_audio(y, sr):
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    tone = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.mean(pitch), tone, tempo


def process_audio(audio_data, sample_rate):# audio processing 
    audio_data = np.frombuffer(audio_data, np.int16).astype(np.float32)
    pitch, tone, tempo = analyze_audio(audio_data, sample_rate)
    return pitch, tone, tempo


# Real-time speech recognition and analysis
def real_time_speech_analysis():
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source)

        while True:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source)
            
            try:
                print("Recognizing speech...")
                text = recognizer.recognize_google(audio)
                print("Text:", text)
                
                if "stop" in text.lower():
                   print("Stopping real-time speech analysis...")
                   break
                
                # Analyze sentiment
                #sentiment = analyze_sentiment(text)
                #print("Sentiment:", sentiment)
                
                # Convert audio to numpy array for analysis
                """audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32)
                sample_rate = recognizer.sample_rate if recognizer.sample_rate is not None else 16000  # Fallback sample rate

                # Analyze audio properties
                pitch, tone, tempo = analyze_audio(audio_data, sample_rate)
                print(f"Pitch: {pitch:.2f}, Tone: {tone:.2f}, Tempo: {tempo:.2f}")
"""
                # Process audio in a separate thread
                sample_rate = recognizer.sample_rate if recognizer.sample_rate is not None else 44000
                
                
                threading.Thread(target=lambda: print(process_audio(audio.get_raw_data(), sample_rate))).start()


            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    except Exception as e:
        print(f"An error occurred during initialization: {e}")

if __name__ == "__main__":
    print("Starting real-time speech analysis...")
    real_time_speech_analysis()

