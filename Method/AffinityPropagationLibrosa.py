
#aqui to usando o librosa p extrair as características
import librosa
import numpy as np
import os
from sklearn.cluster import AffinityPropagation

#coloca o caminho para a a pasta de localizada no seu computador 
audio_folder = ''

# extração de MFCCs 
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.mp3') or f.endswith('.wav')]
features = [extract_features(file) for file in audio_files]


#chamando o affinity propagation
clustering = AffinityPropagation().fit(features)
labels = clustering.labels_

#verificando os clusterss
unique_labels = set(labels)
print(f'Número de clusters: {len(unique_labels)}')

#criando pastas para cada cluster e mover os arquivos de áudio
cluster_folders = [os.path.join(audio_folder, f'cluster_{label}') for label in unique_labels]

for folder in cluster_folders:
    os.makedirs(folder, exist_ok=True)

for file, label in zip(audio_files, labels):
    dest_folder = os.path.join(audio_folder, f'cluster_{label}')
    os.rename(file, os.path.join(dest_folder, os.path.basename(file)))