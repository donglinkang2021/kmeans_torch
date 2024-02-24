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

"""output
X_subvectors: torch.Size([16, 1000, 8])
idx: torch.Size([10])
Iteration 0 mean distance: 3.8903117179870605
Iteration 1 mean distance: 3.2654647827148438
Iteration 2 mean distance: 3.202737331390381
Iteration 3 mean distance: 3.1900131702423096
Iteration 4 mean distance: 3.189864158630371
Iteration 5 mean distance: 3.1919443607330322
Iteration 6 mean distance: 3.194202423095703
Iteration 7 mean distance: 3.1963143348693848
Iteration 8 mean distance: 3.1979026794433594
Iteration 9 mean distance: 3.1993894577026367
Codebooks: torch.Size([16, 10, 8])
Codes: torch.Size([16, 1000])
"""