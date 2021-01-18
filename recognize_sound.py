import os
import json
import math
import librosa
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import load_model
import argparse

SAMPLE_RATE = 22050
TRACK_DURATION = 5 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


classes = [
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping"
]        


def predict(model, X):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    """


    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Predicted label: {}, Confidence: {}".format(classes[predicted_index[0]], prediction[0][predicted_index[0]]))

if __name__ == "__main__":
# construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True,
	help="path to input file")
    ap.add_argument("-m", "--model", type=str,
	default="model.model",
	help="path to trained face mask detector model")
    args = vars(ap.parse_args())
    
    # load audio file
    file_path = args["file"]
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    num_segments = 10
    num_mfcc=13
    n_fft=2048
    hop_length=512

    model = load_model(args["model"])
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        data["mfcc"].append(mfcc.tolist())

    X = np.array(data["mfcc"])
    X = X[..., np.newaxis]
    data_to_predict = X
    predict(model, X)
