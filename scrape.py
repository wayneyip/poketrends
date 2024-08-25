import pokebase as pb
import cv2
from urllib.request import urlopen, Request
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
import random
import webbrowser

# Get pokemon name and number
pokedex_number = random.randrange(1, 1025)
pokemon = pb.pokemon(pokedex_number)
print(pokedex_number, pokemon)

sprite_url = pokemon.sprites.front_default

img_url_request = Request(
	sprite_url, 
	headers={'User-Agent': 'Chrome/23.0.1271.64'}
)
img_url_response = urlopen(img_url_request)

img_webpage = img_url_response.read()
img_array = np.array(bytearray(img_webpage), dtype=np.uint8)
img = cv2.imdecode(img_array, -1)

# Turn all transparent pixels black
img[img[:, :, 3] == 0] = np.zeros(4)

# Convert image to pixel array
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

# Get list of opaque pixels only
opaque_mask = img_rgb[:, :, 3] != 0
opaque_pixels = img_rgb[opaque_mask]

# Cluster the opaque pixels' colors
num_clusters = 5
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(opaque_pixels)
reduced_colors = kmeans.cluster_centers_.astype(int)
reduced_colors_normalized = [color / 255 for color in reduced_colors]

# Reconstruct image with clustered colors
labels = kmeans.labels_
img_compressed = img.copy()
img_compressed[opaque_mask] = reduced_colors[labels]

# Get frequency of each color
unique_labels, counts = np.unique(labels, return_counts=True)
total_pixels = len(labels)
frequencies = counts / total_pixels

# Figure 1: original image
plt.figure(1)
plt.axis("off")
plt.title(f"{pokedex_number}_{pokemon}")
plt.imshow(img_rgb)

# Figure 2: compressed image
plt.figure(2)
plt.axis("off")
plt.title(f"{pokedex_number}_{pokemon} (Compressed)")
plt.imshow(img_compressed)
 
# Figure 3: compressed colors
plt.figure(3)
plt.title(f"{pokedex_number}_{pokemon} Color Distribution")
plt.pie(frequencies, colors=reduced_colors_normalized)

plt.show()