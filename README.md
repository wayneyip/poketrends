# Pokétrends (WIP)
Analyses of Pokémon design trends (color counts, etc) using AI/ML methods.

As a hobbying Pokémon designer, this is my pet project to find underlying patterns across Pokémon designs, and potentially apply them to my own designs (see [@romelle_sprites](https://www.instagram.com/romelle_sprites/)).

## Current Results
- **Design Parameter**: Color count
- **Learning Method**: K-means clustering
- **Cluster Determination**: Elbow method
- **Color Space**: L*A*B

![image](https://github.com/user-attachments/assets/96a2c371-d8ad-4531-9020-d58e9cc8c7bb)

## To-Dos
- Establish better approach for cluster determination
  - Elbow method is still producing unreliable results
- Refine feature scaling for colors
  - Reduced colors are not consistently accurate to original design     
- Improve data preprocessing
  - Black outlines are causing black to be over-represented in the reduced color palette
