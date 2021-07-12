import numpy as np
from drumsamp_models.recommender import get_n_most_similar_sounds
from statistics import mode
from scipy.spatial import distance

def cosine_distance(pathA, pathB):
    a = np.load(pathA).reshape(-1)
    b = np.load(pathB).reshape(-1)

    ml = max(len(a),len(b))
    mfcc = np.concatenate((a , np.zeros(ml-len(a))))
    mfcc_file = np.concatenate((b , np.zeros(ml-len(b))))
    
    return distance.cosine(mfcc, mfcc_file)

def get_custom_tag_nearest(path, tag_dict, cosine_dist_thresh=0.04):
    unique_vals = set(tag_dict.values())
    dists = [np.array([]) for i in range(0, len(unique_vals))]
    distance_dict = dict(zip(unique_vals, dists))

    for p, t in tag_dict.items():
        dist = cosine_distance(path, p)
        distance_dict[t] = np.append(distance_dict[t], dist)

    for t, v in distance_dict.items():
        distance_dict[t] = np.mean(distance_dict[t])
    
    distance_dict = dict(filter(lambda x: x[1] <= cosine_dist_thresh, distance_dict.items()))

    return dict(sorted(distance_dict.items(), key=lambda x: x[1]))

def get_custom_tag(path, sample_library, custom_tag_dict, sound_in_set=False):
    sound_dict = get_n_most_similar_sounds(5, path, sample_library, sound_in_set)
    
    sound_tags = [custom_tag_dict[s] for s in sound_dict.keys()]

    return mode(sound_tags)

def get_custom_tag_mult(paths, sample_library, custom_tag_dict, sound_in_set=False):
    result = {}

    for p in paths:
        result[p] = get_custom_tag(p, sample_library, custom_tag_dict, sound_in_set)
    
    return result

if __name__ == '__main__':
    from pathlib import Path
    tag_dict = {}

    tag_dict['../test_files_recommendation/kick_482056.npy'] = 'distorted'
    tag_dict['../test_files_recommendation/kick_482059.npy'] = 'distorted'
    tag_dict['../test_files_recommendation/kick_482061.npy'] = 'distorted'

    print(get_custom_tag_nearest('../test_files_recommendation/kick_482066.npy', tag_dict))