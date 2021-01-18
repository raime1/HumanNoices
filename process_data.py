"""
Modified version of Valerio Velardo dataser preparation for sound clasiffication
Is modified to use the ESC-50 dataset

Source: https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/12-%20Music%20genre%20classification:%20Preparing%20the%20dataset/code/extract_data.py#L45
ESC-50 dataset: https://github.com/karolpiczak/ESC-50
"""
import json
import os
import math
import librosa
import pandas as pd

#Path to the ESC-50 dataset audio files
DATASET_PATH = "./ESC-50-master/audio"

#JSON to write the processed data
JSON_PATH = "training_data.json"

SAMPLE_RATE = 22050
TRACK_DURATION = 5 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    #Classes of noices made by humans that can be clasified
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

    #Open the metadata file with the info of the audio tracks
    df = pd.read_csv("./ESC-50-master/meta/esc50.csv")

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    #Label index
    i = 0

    # loop through all classes
    for c in classes:
        # save noise label 
        semantic_label = c 
        data["mapping"].append(semantic_label)
        print("\nProcessing: {}".format(semantic_label))

        # process all audio files of the class
        for f in df[df.category.eq(c)].filename:

            # load audio file
            file_path = os.path.join(DATASET_PATH, f)
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

            # process all segments of audio file
            for d in range(num_segments):

                # calculate start and finish sample for current segment
                start = samples_per_segment * d
                finish = start + samples_per_segment

                # extract mfcc
                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T

                # store only mfcc feature with expected number of vectors
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i)
                    print("{}, segment:{}".format(file_path, d+1))
        i = i + 1 

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)