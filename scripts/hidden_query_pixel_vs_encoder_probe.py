#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_fidelity.utils import create_feature_extractor, resolve_feature_layer_for_metric
from torchvision import datasets

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
USE_TRAIN_SPLIT = False
DOWNLOAD = False

N = 32
M = 4
i_star = 1
j_star = 7

tau = 0.5
SEED = 0

ENCODER_NAME = "inception-v3-compat"
ENCODER_IMAGE_SIZE = 299
ENCODER_BATCH = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

assert N >= 2
assert 0 <= i_star < M
assert 0 <= j_star < N

print("device:", DEVICE)
print("repo_root:", REPO_ROOT)
print("data_root:", DATA_ROOT)

mnist = datasets.MNIST(root=str(DATA_ROOT), train=USE_TRAIN_SPLIT, download=DOWNLOAD)
all_images = mnist.data.float().div(255.0)
all_labels = mnist.targets.clone()

perm = torch.randperm(len(all_images), generator=torch.Generator().manual_seed(SEED))
I_R = perm[:N]
I_Q = perm[N : N + M]

R = all_images[I_R].unsqueeze(1)
y_R = all_labels[I_R]
Q = all_images[I_Q].unsqueeze(1)
y_Q = all_labels[I_Q]

q_star = Q[i_star : i_star + 1].clone()
y_star = int(y_Q[i_star].item())
dataset_i_star = int(I_Q[i_star].item())

R_tilde = R.clone()
R_tilde[j_star] = q_star[0]

print()
print("N:", N)
print("M:", M)
print("i_star:", i_star)
print("j_star:", j_star)
print("y_star:", y_star)
print("dataset_i_star:", dataset_i_star)
print("R shape:", tuple(R.shape))
print("Q shape:", tuple(Q.shape))
print("R_tilde shape:", tuple(R_tilde.shape))

Q_pix = Q.flatten(1)
R_pix = R.flatten(1)
R_tilde_pix = R_tilde.flatten(1)

D_pix_clean = torch.cdist(Q_pix, R_pix)
D_pix = torch.cdist(Q_pix, R_tilde_pix)

top2_pix_clean = torch.topk(D_pix_clean, k=2, largest=False, dim=1)
top2_pix = torch.topk(D_pix, k=2, largest=False, dim=1)

d1_pix_clean = top2_pix_clean.values[:, 0]
d2_pix_clean = top2_pix_clean.values[:, 1]
d1_pix = top2_pix.values[:, 0]
d2_pix = top2_pix.values[:, 1]

rho_pix_clean = d1_pix_clean / d2_pix_clean.clamp(min=1.0e-12)
rho_pix = d1_pix / d2_pix.clamp(min=1.0e-12)

j1_pix_clean = top2_pix_clean.indices[:, 0]
j1_pix = top2_pix.indices[:, 0]

f_mem_pix_clean = float((rho_pix_clean < tau).float().mean().item())
f_mem_pix = float((rho_pix < tau).float().mean().item())

pix_df = pd.DataFrame(
    {
        "i": list(range(M)),
        "y_i": y_Q.tolist(),
        "j_1_clean": j1_pix_clean.tolist(),
        "j_1_tilde": j1_pix.tolist(),
        "d_1_clean_pix": d1_pix_clean.tolist(),
        "d_1_tilde_pix": d1_pix.tolist(),
        "d_2_clean_pix": d2_pix_clean.tolist(),
        "d_2_tilde_pix": d2_pix.tolist(),
        "rho_clean_pix": rho_pix_clean.tolist(),
        "rho_tilde_pix": rho_pix.tolist(),
    }
)
pix_df["hit_j_star_pix"] = pix_df["j_1_tilde"] == j_star
pix_df["mem_pix"] = pix_df["rho_tilde_pix"] < tau

D_pix_clean_df = pd.DataFrame(
    D_pix_clean.cpu().numpy(),
    index=[f"q_{i}" for i in range(M)],
    columns=[f"r_{j}" for j in range(N)],
)
D_pix_df = pd.DataFrame(
    D_pix.cpu().numpy(),
    index=[f"q_{i}" for i in range(M)],
    columns=[f"r_{j}" for j in range(N)],
)

ell = resolve_feature_layer_for_metric("fid", feature_extractor=ENCODER_NAME)
phi = create_feature_extractor(
    ENCODER_NAME,
    [ell],
    cuda=DEVICE.startswith("cuda"),
    verbose=False,
)

Q_rgb = Q.repeat(1, 3, 1, 1)
R_rgb = R.repeat(1, 3, 1, 1)
R_tilde_rgb = R_tilde.repeat(1, 3, 1, 1)

Q_rgb = F.interpolate(
    Q_rgb,
    size=(ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE),
    mode="bilinear",
    align_corners=False,
)
R_rgb = F.interpolate(
    R_rgb,
    size=(ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE),
    mode="bilinear",
    align_corners=False,
)
R_tilde_rgb = F.interpolate(
    R_tilde_rgb,
    size=(ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE),
    mode="bilinear",
    align_corners=False,
)

Q_u8 = torch.round(Q_rgb * 255.0).clamp(0, 255).to(torch.uint8)
R_u8 = torch.round(R_rgb * 255.0).clamp(0, 255).to(torch.uint8)
R_tilde_u8 = torch.round(R_tilde_rgb * 255.0).clamp(0, 255).to(torch.uint8)

Phi_Q = []
for start in range(0, len(Q_u8), ENCODER_BATCH):
    out = phi(Q_u8[start : start + ENCODER_BATCH].to(DEVICE))
    feats = phi.convert_features_tuple_to_dict(out)[ell].flatten(1).float().cpu()
    Phi_Q.append(feats)
Phi_Q = torch.cat(Phi_Q, dim=0)

Phi_R = []
for start in range(0, len(R_u8), ENCODER_BATCH):
    out = phi(R_u8[start : start + ENCODER_BATCH].to(DEVICE))
    feats = phi.convert_features_tuple_to_dict(out)[ell].flatten(1).float().cpu()
    Phi_R.append(feats)
Phi_R = torch.cat(Phi_R, dim=0)

Phi_R_tilde = []
for start in range(0, len(R_tilde_u8), ENCODER_BATCH):
    out = phi(R_tilde_u8[start : start + ENCODER_BATCH].to(DEVICE))
    feats = phi.convert_features_tuple_to_dict(out)[ell].flatten(1).float().cpu()
    Phi_R_tilde.append(feats)
Phi_R_tilde = torch.cat(Phi_R_tilde, dim=0)

D_enc_clean = torch.cdist(Phi_Q, Phi_R)
D_enc = torch.cdist(Phi_Q, Phi_R_tilde)

top2_enc_clean = torch.topk(D_enc_clean, k=2, largest=False, dim=1)
top2_enc = torch.topk(D_enc, k=2, largest=False, dim=1)

d1_enc_clean = top2_enc_clean.values[:, 0]
d2_enc_clean = top2_enc_clean.values[:, 1]
d1_enc = top2_enc.values[:, 0]
d2_enc = top2_enc.values[:, 1]

rho_enc_clean = d1_enc_clean / d2_enc_clean.clamp(min=1.0e-12)
rho_enc = d1_enc / d2_enc.clamp(min=1.0e-12)

j1_enc_clean = top2_enc_clean.indices[:, 0]
j1_enc = top2_enc.indices[:, 0]

f_mem_enc_clean = float((rho_enc_clean < tau).float().mean().item())
f_mem_enc = float((rho_enc < tau).float().mean().item())

enc_df = pd.DataFrame(
    {
        "i": list(range(M)),
        "y_i": y_Q.tolist(),
        "j_1_clean": j1_enc_clean.tolist(),
        "j_1_tilde": j1_enc.tolist(),
        "d_1_clean_enc": d1_enc_clean.tolist(),
        "d_1_tilde_enc": d1_enc.tolist(),
        "d_2_clean_enc": d2_enc_clean.tolist(),
        "d_2_tilde_enc": d2_enc.tolist(),
        "rho_clean_enc": rho_enc_clean.tolist(),
        "rho_tilde_enc": rho_enc.tolist(),
    }
)
enc_df["hit_j_star_enc"] = enc_df["j_1_tilde"] == j_star
enc_df["mem_enc"] = enc_df["rho_tilde_enc"] < tau

D_enc_clean_df = pd.DataFrame(
    D_enc_clean.cpu().numpy(),
    index=[f"q_{i}" for i in range(M)],
    columns=[f"r_{j}" for j in range(N)],
)
D_enc_df = pd.DataFrame(
    D_enc.cpu().numpy(),
    index=[f"q_{i}" for i in range(M)],
    columns=[f"r_{j}" for j in range(N)],
)

summary_df = pd.DataFrame(
    {
        "i": list(range(M)),
        "y_i": y_Q.tolist(),
        "j_1_pix": j1_pix.tolist(),
        "rho_pix": rho_pix.tolist(),
        "hit_j_star_pix": (j1_pix == j_star).tolist(),
        "j_1_enc": j1_enc.tolist(),
        "rho_enc": rho_enc.tolist(),
        "hit_j_star_enc": (j1_enc == j_star).tolist(),
    }
)

print()
print("feature layer ell:", ell)
print("Phi(Q) shape:", tuple(Phi_Q.shape))
print("Phi(R_tilde) shape:", tuple(Phi_R_tilde.shape))

print()
print("=== D_pix_clean ===")
print(D_pix_clean_df.to_string(float_format=lambda x: f"{x:.6f}"))
print()
print("=== D_pix ===")
print(D_pix_df.to_string(float_format=lambda x: f"{x:.6f}"))

print()
print("=== Pixel Summary ===")
print(pix_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print()
print(f"f_mem_pix_clean = {f_mem_pix_clean:.6f}")
print(f"f_mem_pix       = {f_mem_pix:.6f}")

print()
print("=== D_enc_clean ===")
print(D_enc_clean_df.to_string(float_format=lambda x: f"{x:.6f}"))
print()
print("=== D_enc ===")
print(D_enc_df.to_string(float_format=lambda x: f"{x:.6f}"))

print()
print("=== Encoder Summary ===")
print(enc_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print()
print(f"f_mem_enc_clean = {f_mem_enc_clean:.6f}")
print(f"f_mem_enc       = {f_mem_enc:.6f}")

print()
print("=== Final Check ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print()
print("target j_star:", j_star)
print("hidden i_star:", i_star)
print("pixel recovered j_star:", int(j1_pix[i_star].item()) == j_star)
print("encoder recovered j_star:", int(j1_enc[i_star].item()) == j_star)
print(f"pixel d_1(i_star)   = {float(d1_pix[i_star].item()):.6f}")
print(f"pixel rho(i_star)   = {float(rho_pix[i_star].item()):.6f}")
print(f"encoder d_1(i_star) = {float(d1_enc[i_star].item()):.6f}")
print(f"encoder rho(i_star) = {float(rho_enc[i_star].item()):.6f}")
