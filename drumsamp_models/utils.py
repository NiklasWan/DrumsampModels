import numpy as np
import librosa
import drumsamp_models.config as config
import os

# Taken from https://github.com/astorfi/speechpy/blob/master/speechpy/processing.py

def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output

def compute_normalized_mel(path):
    y, sr = librosa.load(path)

    features = librosa.feature.mfcc(y=y, sr=sr)
    return cmvn(features, True)

# Taken from https://github.com/Quint-e/musicnn_keras/blob/master/musicnn_keras/configuration.py
def batch_classification_data(audio_file):
    '''For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

    INPUT
    
    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'

    - n_frames: length (in frames) of the input spectrogram patches.
    Data format: integer.
    Example: 187
        
    - overlap: ammount of overlap (in frames) of the input spectrogram patches.
    Note: Set it considering n_frames.
    Data format: integer.
    Example: 10
    
    OUTPUT
    
    - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
    Data format: 3D np.array (batch, time, frequency)
    
    - audio_rep: raw audio representation (spectrogram).
    Data format: 2D np.array (time, frequency)
    '''

    n_frames = librosa.time_to_frames(3, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    overlap = n_frames

    # compute the log-mel spectrogram with librosa
    audio, sr = librosa.load(audio_file, sr=config.SR)

    # audio needs to be min 3 secs long, pad with zeros if it is not
    audio_len = audio.shape[0]
    required_len = sr * 3

    num_pad = required_len - audio_len

    if num_pad > 0:
        if len(audio.shape) > 1:
            audio = np.append(audio, np.zeros((num_pad, audio.shape[1])))
        else:
            audio = np.append(audio, np.zeros((num_pad,)))

    audio_rep = librosa.feature.melspectrogram(y=audio, 
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep

def save_classification_batches_to_disk(input_files, out):
    if not os.path.exists(out):
        os.makedirs(out)

    for f in input_files:
        id = f.split('/')[-1].split('.')[0]
        batch, _ = batch_classification_data(f)

        np.save(os.path.join(out, id + '.npy'), batch)

def save_recommendation_batches_to_disk(input_files, out):
    if not os.path.exists(out):
        os.makedirs(out)

    for f in input_files:
        id = f.split('/')[-1].split('.')[0]
        batch = compute_normalized_mel(f)

        np.save(os.path.join(out, id + '.npy'), batch)


if __name__ == '__main__':
   paths = []
   from pathlib import Path

   paths.extend(Path('../DRUMSAMP/test').glob('**/*.wav'))

   paths = [str(f) for f in paths]

   save_recommendation_batches_to_disk(paths, '../test_files_recommendation')