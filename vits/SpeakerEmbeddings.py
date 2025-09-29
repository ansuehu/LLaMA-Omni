# Load your trained model
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

checkpoint = torch.load("./logs/celtia_gl/G_500000.pth", map_location="cpu")

print(checkpoint.keys())


#speaker_embedding_weights = checkpoint["model"]["emb_g.weight"]  # Shape: (num_speakers, embedding_dim)
speaker_embedding_weights = checkpoint["model"]["emb_g.weight"]

X = speaker_embedding_weights.numpy()
print(x)
exit()
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
X_2d = tsne.fit_transform(X)

speaker_ids = [f"spk_{i}" for i in range(X.shape[0])]

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=range(len(speaker_ids)), cmap='tab20', s=100)

for i, label in enumerate(speaker_ids):
    plt.text(X_2d[i, 0], X_2d[i, 1], label, fontsize=9)

plt.title("Learned Speaker Embeddings (VITS)")
plt.xlabel("TSNE Dim 1")
plt.ylabel("TSNE Dim 2")
plt.grid(True)
plt.savefig('MultiSpeaker_G_4000.png')