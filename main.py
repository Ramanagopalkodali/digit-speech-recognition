import os
import random
import numpy as np
from feature_extract import feature_extract
from models import GMM, VectorQuantizer, HMM
PATH = "recordings" 
CODEBOOK = 128 

def file_parameters(file):
    parts = file.replace('.wav', '').split('_')
    return int(parts[0]), int(parts[1].replace('speaker', ''))
def load_split():
    print("Loading features 80")
    train_data_continuous={d: [] for d in range(10)} 
    train_data_flat={d: [] for d in range(10)} 
    train = [] 
    test_files= []           
    
    for file in os.listdir(PATH):
        if not  file.endswith('.wav'): continue
        filepath = os.path.join(PATH, file)
        digit, speaker = file_parameters(file)

        #automatically seperates the recordings 
        if speaker <= 5:
            train.append((filepath, digit))
        elif speaker == 6:
            test_files.append((filepath, digit))

    random.seed(42) 
    random.shuffle(train)

    # spliting 80% for train and 20% for validation of the train
    split_idx = int(len(train) * 0.8)
    train_files = train[:split_idx]
    val_files = train[split_idx:]
    
    all_training_features_flat = []
    for filepath, digit in train_files:
        features = feature_extract(filepath)
        train_data_continuous[digit].append(features)
        train_data_flat[digit].append(features)
        all_training_features_flat.append(features)
    for d in range(10):
        if len(train_data_flat[d]) > 0:

            train_data_flat[d] = np.vstack(train_data_flat[d])
    all_training_features_flat = np.vstack(all_training_features_flat)
    return train_data_continuous, train_data_flat, all_training_features_flat, val_files, test_files

# TRAINING MODEL
train_cont_data, train_flat_data, all_flat_features, val_files, test_files = load_split()
vq_model = VectorQuantizer(n_clusters=CODEBOOK)
vq_model.fit(all_flat_features)
gmm_models = {}
hmm_models = {}

for digit in range(10):
    #training GMM for individual digits
    gmm = GMM(n_components=8)
    if len(train_flat_data[digit]) > 0:
        gmm.fit(train_flat_data[digit])
    gmm_models[digit] = gmm
    
    #training HMM for individual digits 
    hmm = HMM(n_states=4, n_symbols=CODEBOOK)
    if len(train_cont_data[digit]) > 0:
        discrete_sequences = [vq_model.transform(seq) for seq in train_cont_data[digit]]
        hmm.fit(discrete_sequences) 
    hmm_models[digit] = hmm

def inference(audio_file, model_dict, model_type, vq_model=None):
    scores = {}
    features = feature_extract(audio_file)
    if model_type == "GMM":
        for digit, model in model_dict.items():
            scores[digit] = model.score(features)
    elif model_type == "HMM":
        discrete_seq = vq_model.transform(features)
        for digit, model in model_dict.items():
            scores[digit] = model.score(discrete_seq)
            
    digit = max(scores, key=scores.get)
    return digit

def get_metrics(model_dict, model_type, file_list, vq_model=None):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    correct = 0
    total = len(file_list)
    if total == 0: 
        return 0, confusion_matrix, [0]*10
        
    for filepath, actual_digit in file_list:
        digit = inference(filepath, model_dict, model_type, vq_model)
        confusion_matrix[actual_digit][digit] += 1
        if digit == actual_digit:
            correct += 1
    accuracy =(correct/total) *100
    per_digit_acc = []
    for d in range(10):
        total_d = np.sum(confusion_matrix[d, :])
        correct_d = confusion_matrix[d, d]
        acc_d = (correct_d/total_d) *100 if total_d > 0 else 0
        per_digit_acc.append(acc_d)
    return accuracy, confusion_matrix, per_digit_acc

# TESTING MODELS
gmm_va, gmm_vcm, gmm_val_digit = get_metrics(gmm_models, "GMM", val_files)
gmm_ta, gmm_tcm, gmm_test_digit = get_metrics(gmm_models, "GMM", test_files)
hmm_va, hmm_vcm, hmm_val_digit = get_metrics(hmm_models, "HMM", val_files, vq_model)
hmm_ta, hmm_tcm, hmm_test_digit = get_metrics(hmm_models, "HMM", test_files, vq_model)
# MENTRICS
print("\n Overall accuracies...")
print(f"{'Metric':<30} |{'GMM ':<15} |{'HMM ':<15}")
print(f"{'Validation Accuracy ':<30} |{gmm_va:>6.2f}% |{hmm_va:>6.2f}%")
print(f"{'Test Accuracy (Speaker 6)':<30} |{gmm_ta:>6.2f}% |{hmm_ta:>6.2f}%")
print("\nPer digit accuracy... ")
print(f"{'Digit':<6} |{'GMM Val':<10} |{'GMM Test':<10} |{'HMM Val':<10} |{'HMM Test':<10}")
print("\n")
print("\n")

for d in range(10):
    print(f" {d:<4} |{gmm_test_digit[d]:>7.2f}% |{gmm_test_digit[d]:>7.2f}% |{hmm_val_digit[d]:>7.2f}% |{hmm_test_digit[d]:>7.2f}%")

print("\nGMM Validation Confusion matrix:")
print(gmm_vcm)
print("\nHMM Validation Confusion Matrix:")
print(hmm_vcm)
print("\n")
print("\n")
print("\nGMM test confusion Matrix:")
print(gmm_tcm)
print("\nHMM Test Confusion matrix:")
print(hmm_tcm)
