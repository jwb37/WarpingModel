import torch
import torch.nn.functional as F


def correlation_map(featsA, featsB, flatten=True):
    batch_size = featsA.size(0)

    batch = [
        torch.tensordot(featsA[i], featsB[i], dims=([0],[0]))
        for i in range(batch_size)
    ]

    out = torch.stack(batch, dim=0)
    out = out.flatten(start_dim=3)
    out = F.normalize(out, dim=3)

    if flatten:
        return out.movedim(-1, 1)
    else:
        N, H, W, X = out.size()
        out = out.view(N, H, W, H, W)
        return out
