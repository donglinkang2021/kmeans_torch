import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def product_quantization(X: torch.Tensor, v_clusters: int, g_subvectors: int, num_iterations: int):
    n_vectors, vector_dim = X.shape
    assert vector_dim % g_subvectors == 0, 'vector_dim must be divisible by g_subvectors'
    X_subvectors = rearrange(X, 'n (g d) -> g n d', g=g_subvectors)

    # choose random codebooks from the data
    idx = torch.randint(n_vectors, (v_clusters,), dtype=torch.long)
    codebooks = X_subvectors[:, idx] # g, v, d

    eps = 1e-8
    for _ in range(num_iterations):
        distances = torch.cdist(X_subvectors, codebooks, p=2) # g n d, g v d -> g n v
        codes = distances.argmin(dim=-1) # g n v -> g n
        # # turn codes into one-hot
        codes_onehot = F.one_hot(codes, v_clusters).float().transpose(1, 2) # g n -> g v n
        # # update codebooks
        codebooks = (codes_onehot @ X_subvectors) / (codes_onehot.sum(dim=-1, keepdim=True) + eps) # g v n @ g n d -> g v d

    return codebooks, codes

n_vectors = 1000
vector_dim = 128
X = torch.randn(n_vectors, vector_dim)

pq_kwargs = {
    'v_clusters': 256,    # number of clusters
    'g_subvectors': 16,   # groups subvectors
    'num_iterations': 20, # number of iterations
}

codebooks, codes = product_quantization(X, **pq_kwargs)
print("Codebooks:", codebooks.shape)  # (g_subvectors, v_clusters, vector_dim / g_subvectors)
print("Codes:", codes.shape)  # (g_subvectors, n_vectors)

# use the codebooks and codes to quantize the input
# here we will replace the codes ith row jth cols element c value with the c column the codebooks
codes_onehot = F.one_hot(codes, pq_kwargs['v_clusters']).float() # g n -> g n v
X_quantized = codes_onehot @ codebooks # g n v @ g v d -> g n d
X_quantized = rearrange(X_quantized, 'g n d -> n (g d)')
print("Quantized X:", X_quantized.shape)
print("Original X:", X.shape)
print("Quantized X[0]:", X_quantized[0])
print("Original X[0]:", X[0])

# check if the quantized X is close to the original X
# mseloss
loss = F.mse_loss(X, X_quantized)
print("MSE Loss:", loss.item())

# calculate the compression ratio
# compression ratio % version
original_size = n_vectors * vector_dim
compressed_size = pq_kwargs['g_subvectors'] * pq_kwargs['v_clusters'] * (vector_dim / pq_kwargs['g_subvectors']) + pq_kwargs['g_subvectors'] * n_vectors
compression_ratio = compressed_size / original_size
print(f"Compression Ratio: {compression_ratio * 100:.2f}%")

"""
Codebooks: torch.Size([16, 256, 8])
Codes: torch.Size([16, 1000])
Quantized X: torch.Size([1000, 128])
Original X: torch.Size([1000, 128])
Quantized X[0]: tensor([-0.7868,  0.6093, -0.2525, -0.2337, -0.9972, -1.5672, -1.1376, -1.1332,
        -0.5829,  0.6936, -1.1660,  0.1734, -0.2531, -0.4533, -0.5873, -0.5144,
         1.7968,  0.0821,  0.6172,  0.5257, -0.1796,  0.1946,  0.2233, -0.8959,
         0.4537, -0.0966, -0.5252, -0.6704,  0.2014,  0.2873, -0.0073, -1.6488,
        -0.2808,  0.8886,  1.5606,  1.0118, -0.2515,  0.3511,  0.9616, -0.6199,
        -1.2077,  0.4653,  0.2484, -0.8279,  0.8907,  0.2433,  0.4322,  0.8314,
         0.5574,  0.9982,  0.0026,  0.1671,  0.7751, -0.9775,  1.7817, -1.8902,
        -0.6293,  0.4910, -0.6331,  0.4258, -0.5796, -0.1664, -0.3136, -0.4892,
        -0.4923,  0.2533,  1.1611,  0.8662, -0.5413, -0.0245, -1.4310,  0.2450,
         1.4039,  0.7114,  0.5089, -0.7030, -0.0104, -0.1743,  1.7894, -0.6944,
        -0.7205,  0.0307, -0.4830, -1.2859, -1.3215, -1.0200,  0.9091,  0.7179,
        -0.8483,  0.4271, -0.8934, -0.6717, -1.2070, -0.2711,  0.5146,  1.0029,
        -0.7669, -0.0588,  0.1296, -0.4571, -1.5900, -0.0063,  0.0735, -0.2059,
        -0.0955,  0.3661,  0.6004, -1.2144, -1.4699,  0.6223, -0.0322, -1.3047,
        -0.2139,  0.0228,  0.4717,  0.8528, -1.9234, -0.6760,  0.6359,  2.1142,
         1.3455, -0.4977,  1.8387,  0.2118,  0.0521,  0.5181, -0.5803, -0.7178])
Original X[0]: tensor([-1.8027,  0.4725,  0.0422, -0.4757, -0.5632, -2.0128, -0.9776, -1.3719,
        -0.8285,  0.7473, -1.8298,  0.2277, -0.6246, -0.8422, -1.6290, -0.7707,
         2.2134, -0.1398,  1.0777, -0.0608, -0.9864,  0.6184,  0.3978, -1.1608,
         1.0881,  0.3343,  0.0861, -0.5895, -0.4206,  0.1505, -0.0352, -2.0358,
        -0.4302,  1.6573,  2.8589,  1.6300,  0.9692,  0.5042,  0.3685, -1.1621,
        -0.8038,  0.7519,  0.5642, -0.3883,  0.7939,  0.7292, -0.1476,  1.2536,
         0.8137,  1.3168,  0.2568, -0.4701,  0.8604, -1.4444,  1.4209, -1.3756,
        -0.2325,  0.0206, -0.2560,  0.6260, -0.6686,  0.3189, -0.8758, -1.1865,
        -1.1600,  0.0577,  1.1186,  0.4216, -0.7234, -0.0270, -2.7811,  1.1859,
         1.3772,  0.7936, -0.1560, -0.8584,  0.3726, -0.1123,  2.5538, -0.3886,
        -1.2232,  0.3810,  0.0209, -1.4287, -1.4145, -1.3061,  1.3238,  0.4262,
        -0.6606, -0.0636, -0.9961, -1.1328, -1.8160, -1.3573,  0.6377,  1.3644,
        -0.2530,  0.3824,  0.3810, -0.1723, -2.0388, -0.4081,  0.7175,  0.0761,
        -0.7041,  0.5171, -0.1330, -1.3361, -1.0337, -0.1510,  0.1781, -1.5275,
        -0.4679,  0.5962, -0.1743,  0.1977, -3.1570, -1.5997,  0.4936,  2.3354,
         0.4932, -0.0334,  2.0084,  0.5135, -0.8743,  0.1315, -1.8184, -1.2188])
MSE Loss: 0.20071998238563538
Compression Ratio: 38.10%
"""
