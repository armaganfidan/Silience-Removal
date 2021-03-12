import librosa
import os
import numpy as np
import soundfile as sf

DATASET_PATH = r"C:\Users\armag\Desktop\dataset"
SAMPLE_RATE = 22050
NEW_DATASET_PATH = r"C:\Users\armag\Desktop\NEWDATASET"


def rms(signal, hop_length=512, frame_size=1024):

    rmse = []
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length):
        rmse_current_frame = np.sqrt(sum(signal[i:i + frame_size] ** 2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)


def silience_removal(signal, rms_signal, zcr_signal,sample_rate=22050, hop_length=512, frame_size=1024):

    new_frame_size = len(signal) // hop_length
    new_max_sample_size = new_frame_size * hop_length
    signal = signal[0:new_max_sample_size]

    new_signal = []
    if len(zcr_signal) == len(rms_signal):
        for i in range(0, len(rms_signal)):
            if np.abs(rms_signal[i]) >= 0.005 and zcr_signal[i] <= 20:
                for element in signal[i * hop_length:(i + 1) * hop_length]:
                    new_signal.append(element)

    else:

        for i in range(0, len(rms_signal)):
            if np.abs(rms_signal[i]) >= 0.005:
                for element in signal[i * hop_length:(i + 1) * hop_length]:
                    new_signal.append(element)
    return new_signal



def creating_new_file(signal, file_name, new_dateset_path):

    if not os.path.exists(new_dateset_path):
        os.makedirs(new_dateset_path)
    new_file_name = file_name.split(".")[0]
    new_file_name = new_dateset_path + str("\\") + new_file_name + str(".wav")
    print(new_file_name)
    new_sound = sf.write(new_file_name, signal, 22050, 'PCM_24')
    print(new_file_name.split())


def unvoiced_to_voiced_signals(datapath, new_data_set,sample_rate=22050, hop_length=512, frame_size=1024):

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(datapath)):

        if dirpath is not datapath:
            labels = dirpath.split("\\")[-1]
            data_set_file = NEW_DATASET_PATH + str("\\") + str(labels)

            for file in filenames:

                file_path = os.path.join(dirpath, file)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = np.array(signal)

                zcr_signal = librosa.feature.zero_crossing_rate(signal)

                rms_signal = rms(signal)
                new_sig = silience_removal(signal, rms_signal=rms_signal, zcr_signal=zcr_signal)
                new_sig = np.array(new_sig, dtype=float)

                creating_new_file(new_sig, file, data_set_file)

if __name__ == "__main__":
    unvoiced_to_voiced_signals(DATASET_PATH, NEW_DATASET_PATH)




