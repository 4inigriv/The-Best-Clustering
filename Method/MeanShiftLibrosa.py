
#Testei o algoritmo utilizando Librosa, para ver se possui diferença nos resultados!
import os
import shutil
import numpy as np
import librosa
from sklearn.cluster import MeanShift

#caminho para a pasta do seu computador que possui os audios 
pasta_audio =  ''

# lista todos os arquivos de áudio na pasta
arquivos_audio = [os.path.join(pasta_audio, f) for f in os.listdir(pasta_audio) if f.endswith('.wav')]

# função para extrair características dos áudios
def extrair_caracteristicas(arquivo):
    y, sr = librosa.load(arquivo)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

caracteristicas = np.array([extrair_caracteristicas(f) for f in arquivos_audio])

# chamando o algoritmo MeanShift
ms = MeanShift()
ms.fit(caracteristicas)
labels = ms.labels_

#pastas para os clusters
for label in np.unique(labels): #de preferencia que só tenha audios nessa pasta
    pasta_cluster = os.path.join(pasta_audio, f'cluster_{label}')
    os.makedirs(pasta_cluster, exist_ok=True)
    for i, arquivo in enumerate(arquivos_audio):
        if labels[i] == label:
            shutil.move(arquivo, os.path.join(pasta_cluster, os.path.basename(arquivo)))

print("clustering concluído!")
