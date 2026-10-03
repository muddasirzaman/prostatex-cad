"""Pretrained ResNet-18 fine-tuned over the SAME 25 splits used by the MATLAB models.
Usage:  python train_resnet_cv.py 1     (1 seed = about 5 folds; use 5 for the full run)
"""
import sys, time
import numpy as np
import scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score

DATA = r"D:\prostate cance\study\dataset_python.mat"
OUT  = r"D:\prostate cance\study\results\resnet_oof.mat"
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EPOCHS, IMG, BATCH, LR = 15, 128, 16, 1e-4     # fixed in advance; do not tune on results

d = sio.loadmat(DATA)
X = np.transpose(d['X'].astype(np.float32), (3, 2, 0, 1))   # N, 3, 64, 64
y = d['y'].ravel().astype(int)
foldOf = d['foldOf'].astype(int)                              # N x nSeeds
n = len(y)
print(f"Loaded {n} lesions ({y.sum()} positive), using {N_SEEDS} seed(s)", flush=True)

def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)   # ImageNet weights
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 2))
    return m

def augment(xb):
    out = []
    for x in xb:
        if torch.rand(1) < 0.5: x = torch.flip(x, [2])
        if torch.rand(1) < 0.5: x = torch.flip(x, [1])
        x = torch.rot90(x, int(torch.randint(0, 4, (1,))), [1, 2])
        out.append(x)
    return torch.stack(out)

def prep(xb):
    return F.interpolate(xb, size=(IMG, IMG), mode='bilinear', align_corners=False)

def run_fold(Xtr, ytr, Xva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr[:, 1].mean(), Xtr[:, 1].std() + 1e-6      # ADC standardised on TRAIN only
    Xtr = Xtr.copy(); Xva = Xva.copy()
    Xtr[:, 1] = (Xtr[:, 1] - mu) / sd
    Xva[:, 1] = (Xva[:, 1] - mu) / sd
    Xtr_t, ytr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xva)

    model = make_model()
    w = torch.tensor([len(ytr) / (2.0 * (ytr == 0).sum()),
                      len(ytr) / (2.0 * (ytr == 1).sum())], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2: continue                        # BatchNorm needs >= 2
            xb = prep(augment(Xtr_t[idx]))
            opt.zero_grad()
            loss = crit(model(xb), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(Xva_t), 32):
            probs.append(torch.softmax(model(prep(Xva_t[i:i + 32])), 1)[:, 1].numpy())
    return np.concatenate(probs)

oof = np.full((n, N_SEEDS), np.nan)
t0 = time.time()
for s in range(N_SEEDS):
    for f in range(1, 6):
        va = foldOf[:, s] == f
        oof[va, s] = run_fold(X[~va], y[~va], X[va], seed=100 * s + f)
        print(f"  seed {s+1} fold {f} done   ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

aucs = [roc_auc_score(y, oof[:, s]) for s in range(N_SEEDS)]
print("\n================= RESULT =================")
print(f"Pretrained ResNet-18, {N_SEEDS} seed(s) x 5 folds, {EPOCHS} epochs, {n} lesions ({y.sum()} positive)")
print(f"ResNet pooled AUC : {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1) if N_SEEDS > 1 else 0.0:.3f}   {[round(a, 3) for a in aucs]}")
print("Reference         : ADC-only 0.740 | Logistic 0.752 | RF 0.765 | small CNN 0.797")
print(f"Total time        : {(time.time()-t0)/60:.1f} minutes")
print("==========================================")
sio.savemat(OUT, {'oof': oof, 'y': y.reshape(-1, 1)})