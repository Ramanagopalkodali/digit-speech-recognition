import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def feature_extract(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)


    y = np.append(y[0],y[1:] -0.97 * y[:-1])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    delta_mfcc = librosa.feature.delta(mfcc, width=3)

    features = np.concatenate((mfcc, delta_mfcc), axis=0)
    features = features.T
    features -= np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    features /= (std_dev + 1e-8) 
    return features