
#importante instalar o essentia no seu computador para o algoritmo funcionar, é ele que utilizo para extrair 
#características

import essentia
import essentia.standard as es
import numpy as np 
import os
from sklearn.cluster import AffinityPropagation

# Caminho para a pasta com audios dentro dela, podendo ser mp3 ou tbm em formato wav, tanto faz
audio_folder = ''


def extract_features(file_path):
    loader = es.MonoLoader(filename=file_path)
    audio = loader()
    #extraçao do mfcc
    mfcc_extractor = es.MFCC()
    spectrum = es.Spectrum()
    window = es.Windowing(type='hann')
    mfccs = []

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)
        mfcc_bands, mfcc_coeffs = mfcc_extractor(frame_spectrum)
        mfccs.append(mfcc_coeffs)

    mfccs = np.array(mfccs)
    return np.mean(mfccs, axis=0) #media do mfccs p n ficar tao grande
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.mp3') or f.endswith('.wav')]
features = [extract_features(file) for file in audio_files]

# chamando o Affinity Propagation
clustering = AffinityPropagation().fit(features)
labels = clustering.labels_

# verificação dos clusters
unique_labels = set(labels)
print(f'Número de clusters: {len(unique_labels)}')

#criaçao de pasta
cluster_folders = [os.path.join(audio_folder, f'cluster_{label}') for label in unique_labels]

for folder in cluster_folders:
    os.makedirs(folder, exist_ok=True)

for file, label in zip(audio_files, labels):
    dest_folder = os.path.join(audio_folder, f'cluster_{label}')
    os.rename(file, os.path.join(dest_folder, os.path.basename(file)))
