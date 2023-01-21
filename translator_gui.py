import streamlit as st
import numpy as np
from PIL import Image
from joblib import load
import umap
import matplotlib.pyplot as plt
import math

from autoencoder import Autoencoder
plt.style.use('dark_background')

st.write("""
# Winnie translator
""")

level = st.radio('Choose set difficulty level', ['hard', 'die hard', '💀'], horizontal=True)

if level == 'hard':
    level_num = 1
elif level == 'die hard':
    level_num = 2
elif level == '💀':
    level_num = 3

file = st.file_uploader('Pick a page to translate')

if file:
    
    kmnist_page = Image.open(file)

    st.write("## Uploaded page:")
    st.image(kmnist_page)

    kmnist_data = []

    letter_size = 32
    letters_per_width = 80
    letters_per_height = 114

    where_blank = []

    k_page = np.array(kmnist_page)[:,:,0]/255

    for l_h in range(letters_per_height):
        for l_w in range(letters_per_width):
            if sum(sum(k_page[l_h * letter_size:(l_h + 1) * letter_size,l_w * letter_size:(l_w + 1) * letter_size])) > 80:
                kmnist_data.append(k_page[l_h * letter_size:(l_h + 1) * letter_size,l_w * letter_size:(l_w + 1) * letter_size])
            else:
                where_blank.append([l_h, l_w])
                
    kmnist_data = np.array(kmnist_data)

    kmnist_autoencoder =  Autoencoder(5)
    emnist_autoencoder =  Autoencoder(5)

    kmnist_autoencoder.load_weights(f'./models/dense-kmnist-{level_num}/weigths')
    emnist_autoencoder.load_weights(f'./models/dense-emnist-{level_num}/weigths')

    kgmm = load(f'./models/gmm/kgmm-{level_num}.joblib')
    egmm = load(f'./models/gmm/egmm-{level_num}.joblib')

    mapping = np.load('./models/mapping.npy',allow_pickle='TRUE').item()

    k_encoded = kmnist_autoencoder.encoder(kmnist_data).numpy()
    k_labels = kgmm.predict(k_encoded)

    e_labels = []
    for label in k_labels:
        e_labels.append(mapping[label])

    e_coded = egmm.means_[e_labels]
    e_mapped = emnist_autoencoder.decoder(e_coded).numpy()

    page = []

    blank_space = np.full((32, 32), 0)

    k_page = np.array(kmnist_page)[:,:,0]/255

    letter_id = 0
    for l_h in range(letters_per_height):
        new_line = []
        for l_w in range(letters_per_width):
            if [l_h, l_w] not in where_blank:
                new_line.append(e_mapped[letter_id])
                letter_id += 1
            else:
                new_line.append(blank_space)
        new_line = np.concatenate(new_line, axis=1) 
        page.append(new_line)
    page = np.concatenate(page, axis=0) 

    page = page * 255
    page[page > 255/2] = 255
    page[page <= 255/2] = 0

    st.write("## Translation:")
    st.image(page, clamp=True)

    st.write('## Kmnist-emnist mapping:')

    k_pred = kmnist_autoencoder.decoder(kgmm.means_[list(mapping.keys())])
    e_pred = emnist_autoencoder.decoder(egmm.means_[list(mapping.values())])

    k_len = len(list(mapping.keys()))
    width = 10
    height = math.ceil(k_len / 10)
    fig, ax = plt.subplots(height * 2, width, figsize=(20,18))
    

    for h in range(0, height * 2, 2):
        for w in range(width):
            if w + h/2 * width < k_len:
                ax[h, w].imshow(k_pred[w + int(h/2) * width])
                ax[h + 1, w].imshow(e_pred[w + int(h/2) * width])
            else:
                ax[h, w].imshow(blank_space)
                ax[h + 1, w].imshow(blank_space)
            ax[h, w].axis('off')
            ax[h + 1, w].axis('off')

    st.pyplot(fig)

    st.write('## UMAP for kmnist:')
    fig, ax = plt.subplots(figsize=(20,20))

    reducer = umap.UMAP()
    mapper = reducer.fit(k_encoded)
    cmap = plt.get_cmap('RdBu', k_len )
    sct = plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=k_labels, cmap=cmap, s=50, vmin= - 0.5, vmax= k_len + 0.5)
    plt.colorbar(sct, ticks=np.arange(0, k_len + 1))
    plt.gca().set_aspect('equal', 'datalim')
    st.pyplot(fig)
