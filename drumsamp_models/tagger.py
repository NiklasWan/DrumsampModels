from drumsamp_models.utils import batch_classification_data
import drumsamp_models.config as config
import numpy as np
import tensorflow as tf
import pathlib
import os

def predict_from_batch(model, labels, batch, threshold):
    predictions = model.predict_on_batch(batch)
    tags_likelihood_mean = np.mean(predictions, axis=0)

    filtered_likelihood_mean = [el for el in tags_likelihood_mean if el>threshold]
    
    if len(filtered_likelihood_mean) == 0:
        return 'unknown'
    else:
        idx = tags_likelihood_mean.tolist().index(max(tags_likelihood_mean))
        return labels[idx]

def predict_tags_on_raw_audio(file_names, unknown_threshold=0.60):
    tags = {}
    p = os.path.join(pathlib.Path(__file__).parent.absolute(), 'models/DRUMSAMP_MTT_musicnn.hdf5')
    model = tf.keras.models.load_model(p)

    for f in file_names:
        batch, _ = batch_classification_data(f)
        batch = tf.expand_dims(batch, 3)
        tag = predict_from_batch(model, config.DRUMSAMP_LABELS, batch, unknown_threshold)
        tags[f] = tag

    return tags

def predict_tags_on_computed_mels(file_names, unknown_threshold=0.60):
    tags = {}
    p = os.path.join(pathlib.Path(__file__).parent.absolute(), 'models/DRUMSAMP_MTT_musicnn.hdf5')
    model = tf.keras.models.load_model(p)

    for f in file_names:
        batch = np.load(f)
        batch = tf.expand_dims(batch, 3)
        tag = predict_from_batch(model, config.DRUMSAMP_LABELS, batch, unknown_threshold)
        tags[f] = tag
        tf.keras.backend.clear_session()

    return tags
 

if __name__ == '__main__':
    from pathlib import Path
    paths = []
    paths.extend(Path('../test_files').glob('**/*.npy'))
    paths = [str(f) for f in paths]

    tags = predict_tags_on_computed_mels(paths)
    print(tags)