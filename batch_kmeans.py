import torch
import torch.nn.functional as F
from einops import rearrange, repeat

n_vectors = 1000
vector_dim = 128
X = torch.randn(n_vectors, vector_dim)

# product_quantization demo
v_clusters = 10  # number of clusters
g_subvectors = 16 # groups subvectors
num_iterations = 10

n_vectors, vector_dim = X.shape
X_subvectors = rearrange(X, 'n (g d) -> g n d', g=g_subvectors)
print("X_subvectors:", X_subvectors.shape)

# choose random codebooks from the data
idx = torch.randint(n_vectors, (v_clusters,), dtype=torch.long)
print("idx:", idx.shape) # v,
codebooks = X_subvectors[:, idx] # g, v, d


for iteration in range(num_iterations):
    # distances = X_subvectors.unsqueeze(2) - codebooks.unsqueeze(1) # g n 1 d - g 1 v d -> g n v d
    # distances = distances.pow(2).sum(dim=-1) # g n v d -> g n v
    distances = torch.cdist(X_subvectors, codebooks, p=2) # g n d, g v d -> g n v
    codes = distances.argmin(dim=-1) # g n v -> g n
    # turn codes into one-hot
    codes_onehot = F.one_hot(codes, v_clusters).float().transpose(1, 2) # g n -> g v n
    # update codebooks
    codebooks = (codes_onehot @ X_subvectors) / codes_onehot.sum(dim=-1, keepdim=True) # g v n @ g n d -> g v d
    print(f"Iteration {iteration} mean distance: {distances.mean()}")

print("Codebooks:", codebooks.shape)  # (g_subvectors, v_clusters, vector_dim / g_subvectors)
print("Codes:", codes.shape)  # (n_vectors, g_subvectors)
