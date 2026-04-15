import torch


def _flatten(tensor):
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(1)


def nearest_neighbor_stats(samples, reference, chunk_size=256):
    samples = _flatten(samples).float()
    reference = _flatten(reference).float()

    if reference.shape[0] < 2:
        raise ValueError("Memorization metrics require at least two reference samples.")

    d1_list = []
    d2_list = []
    i1_list = []

    for start in range(0, samples.shape[0], chunk_size):
        chunk = samples[start : start + chunk_size]
        distances = torch.cdist(chunk, reference)
        topk = torch.topk(distances, k=2, largest=False, dim=1)
        d1_list.append(topk.values[:, 0])
        d2_list.append(topk.values[:, 1])
        i1_list.append(topk.indices[:, 0])

    d1 = torch.cat(d1_list)
    d2 = torch.cat(d2_list)
    i1 = torch.cat(i1_list)
    ratios = d1 / d2.clamp(min=1.0e-12)

    return {
        "d1": d1,
        "d2": d2,
        "ratios": ratios,
        "indices": i1,
    }


def compute_memorization_metrics(samples, reference, ratio_threshold=0.5, chunk_size=256):
    stats = nearest_neighbor_stats(samples, reference, chunk_size=chunk_size)
    ratios = stats["ratios"]
    return {
        "f_mem": float((ratios < ratio_threshold).float().mean().item()),
        "mean_nn_ratio": float(ratios.mean().item()),
        "mean_d1": float(stats["d1"].mean().item()),
        "mean_d2": float(stats["d2"].mean().item()),
    }
