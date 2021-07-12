import numpy as np
from scipy.spatial import distance
from pathlib import Path
import heapq

def get_n_most_similar_sounds(n, path, sample_library, sound_in_set=False):
    n = n + 1
    files = []
    files.extend(Path(sample_library).glob('**/*.npy'))

    mfcc = np.load(path).reshape(-1)
    min_scores = [np.Infinity for i in range(n)]
    paths = ['' for i in range(n)]
    
    for f in files:
        with open(str(f), mode='rb') as numpy_file:
            mfcc_file = np.load(numpy_file).reshape(-1)
            ml = max(len(mfcc),len(mfcc_file))
            mfcc = np.concatenate((mfcc , np.zeros(ml-len(mfcc))))
            mfcc_file = np.concatenate((mfcc_file , np.zeros(ml-len(mfcc_file))))
            
            dist = distance.cosine(mfcc, mfcc_file)

            current_max = max(min_scores)
            current_max_idx = min_scores.index(current_max)

            if dist < current_max:
                name = str(f).split('/')[-1]

                del min_scores[current_max_idx]
                del paths[current_max_idx]

                min_scores.append(dist)
                paths.append(name)

    result = sorted(dict(zip(paths, min_scores)).items(), key=lambda x: x[1])

    if (sound_in_set):
        return dict(result[1:])
    else:
        return dict(result)

def get_n_most_similar_sounds_mult(n, paths, sample_library, sound_in_set=False):
    results = {}

    for p in paths:
        results.update(get_n_most_similar_sounds(5, p, sample_library, sound_in_set))
    
    return list(sorted(heapq.nsmallest(n, results.items(), key=lambda x: x[1]), key=lambda x: x[1]))

if __name__ == '__main__':
    print(get_n_most_similar_sounds_mult(8, ['/workspaces/drumsamp-models/test_files_recommendation/kick_482066.npy'], '../test_files_recommendation', True))