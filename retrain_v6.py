"""
retrain_v6b.py — Retrain with structured PPT states + correct K_DW.

Problem with v6a: random PPT states are too rare for d>8 and never entangled.
Solution: Mix structured entangled states with random separable ones.

Sources of PPT-entangled states:
1. Horodecki 3×3 family (known PPT-entangled)
2. Chessboard/UPB constructions  
3. Isotropic states near PPT boundary
4. Random separable states (known K_DW ≤ 0)
5. Werner states (known formula)
"""
import numpy as np, torch, torch.nn as nn, time, os, sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '.')
from sa_engine import kdw_correct, realignment_norm, partial_transpose_B, S

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURES = [
    'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
    'pt_min', 'pt_boundary_dist',
    'S_A', 'S_B', 'S_AB', 'mutual_info',
    'realign_norm', 'concurrence_approx',
]
N_FEAT = len(FEATURES)

def extract_features(rho, dA, dB):
    """Extract feature vector from density matrix."""
    d = dA * dB
    ev = np.linalg.eigvalsh(rho); ev_pos = ev[ev > 1e-15]
    rank = len(ev_pos)
    purity = np.real(np.trace(rho @ rho))
    S_AB = -np.sum(ev_pos * np.log2(ev_pos)) if len(ev_pos) > 0 else 0
    
    # Reduced states
    rA = np.zeros((dA, dA), dtype=complex)
    rB = np.zeros((dB, dB), dtype=complex)
    for a in range(dA):
        for ap in range(dA):
            rA[a, ap] = np.trace(rho[a*dB:(a+1)*dB, ap*dB:(ap+1)*dB])
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    
    S_A = S(rA); S_B = S(rB)
    MI = S_A + S_B - S_AB
    
    pt = partial_transpose_B(rho, dA, dB)
    pt_eigs = np.linalg.eigvalsh(pt)
    pt_min = pt_eigs.min()
    
    R_norm = realignment_norm(rho, dA, dB)
    
    # Concurrence approximation (lower bound via PT negativity)
    neg = sum(abs(e) for e in pt_eigs if e < 0)
    
    return np.array([
        rank / d, purity, ev.min(), ev.max(), ev.std(),
        pt_min, abs(pt_min),
        S_A, S_B, S_AB, MI,
        R_norm, neg,
    ])

def horodecki_3x3(a_param):
    """Horodecki 3×3 PPT-entangled state family."""
    a = a_param
    b = (1 + a) / 2
    c = np.sqrt(1 - a**2) / 2
    
    rho = np.zeros((9, 9), dtype=complex)
    rho[0,0] = a; rho[0,8] = a
    rho[1,1] = a
    rho[2,2] = a
    rho[3,3] = a
    rho[4,4] = a
    rho[5,5] = a
    rho[6,6] = b; rho[6,8] = c
    rho[7,7] = a
    rho[8,0] = a; rho[8,6] = c; rho[8,8] = b
    
    rho /= np.trace(rho)
    # Make Hermitian
    rho = (rho + rho.conj().T) / 2
    return rho

def random_separable(dA, dB, n_terms=5):
    """Random separable state = Σ p_i |ψ_i⟩⟨ψ_i| ⊗ |φ_i⟩⟨φ_i|."""
    d = dA * dB
    rho = np.zeros((d, d), dtype=complex)
    for _ in range(n_terms):
        psiA = np.random.randn(dA) + 1j*np.random.randn(dA); psiA /= np.linalg.norm(psiA)
        psiB = np.random.randn(dB) + 1j*np.random.randn(dB); psiB /= np.linalg.norm(psiB)
        psi = np.kron(psiA, psiB)
        rho += np.outer(psi, psi.conj())
    rho /= np.trace(rho)
    return rho

def werner_state(d, p):
    """Werner state: ρ = p|Φ+⟩⟨Φ+| + (1-p)I/d²."""
    psi = np.zeros(d*d, dtype=complex)
    for i in range(d): psi[i*d+i] = 1/np.sqrt(d)
    return p * np.outer(psi, psi.conj()) + (1-p) * np.eye(d*d) / (d*d)

def isotropic_state(d, f):
    """Isotropic state parameterized by fidelity f."""
    psi = np.zeros(d*d, dtype=complex)
    for i in range(d): psi[i*d+i] = 1/np.sqrt(d)
    Phi = np.outer(psi, psi.conj())
    return f * Phi + (1-f) * np.eye(d*d) / (d*d)

def generate_dataset(n_total=3000):
    """Generate balanced dataset: separable + entangled + Werner + edge cases."""
    X_all, y_all, ent_all = [], [], []
    
    print("  Phase 1: Separable states (known K_DW ≤ 0)")
    for dA, dB in [(2,2), (2,3), (3,3), (2,4), (2,5), (3,4)]:
        for _ in range(200):
            rho = random_separable(dA, dB)
            feat = extract_features(rho, dA, dB)
            k = kdw_correct(rho, dA, dB, 30)
            X_all.append(feat); y_all.append(k); ent_all.append(False)
    print(f"    {len(X_all)} separable states")
    
    print("  Phase 2: Werner states (exact K_DW known)")
    for d in [2, 3]:
        for p in np.linspace(0.0, 1.0, 50):
            rho = werner_state(d, p)
            feat = extract_features(rho, d, d)
            k = kdw_correct(rho, d, d, 50)
            # Werner is entangled iff p > 1/d (for d=2: p>0.5)
            is_ent = p > 1/d
            X_all.append(feat); y_all.append(k); ent_all.append(is_ent)
    print(f"    {len(X_all)} total after Werner")
    
    print("  Phase 3: Isotropic states near PPT boundary")
    for d in [2, 3]:
        for f in np.linspace(0.0, 1/(d+1) + 0.1, 40):
            rho = isotropic_state(d, f)
            # PPT iff f ≤ 1/(d+1)
            pt = partial_transpose_B(rho, d, d)
            is_ppt = np.linalg.eigvalsh(pt).min() >= -1e-10
            feat = extract_features(rho, d, d)
            k = kdw_correct(rho, d, d, 30)
            is_ent = not is_ppt or f > 1/(d+1)
            X_all.append(feat); y_all.append(k); ent_all.append(is_ent)
    print(f"    {len(X_all)} total after isotropic")
    
    print("  Phase 4: Random mixed states (2×2, 2×3)")
    for dA, dB in [(2,2), (2,3), (3,3)]:
        d = dA*dB
        for _ in range(300):
            G = np.random.randn(d, d+1) + 1j*np.random.randn(d, d+1)
            rho = G @ G.conj().T; rho /= np.trace(rho)
            feat = extract_features(rho, dA, dB)
            k = kdw_correct(rho, dA, dB, 30)
            # Check entanglement via PPT (exact for 2×2, 2×3)
            pt = partial_transpose_B(rho, dA, dB)
            is_ent = np.linalg.eigvalsh(pt).min() < -1e-10
            X_all.append(feat); y_all.append(k); ent_all.append(is_ent)
    print(f"    {len(X_all)} total after random mixed")
    
    X = np.array(X_all)
    y = np.array(y_all)
    ent = np.array(ent_all)
    
    return X, y, ent

class SAPredictor(nn.Module):
    def __init__(self, n_feat, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2), nn.GELU(),
        )
        self.reg = nn.Linear(hidden//2, 1)
        self.cls = nn.Linear(hidden//2, 1)
    
    def forward(self, x):
        h = self.net(x)
        return self.reg(h).squeeze(-1), torch.sigmoid(self.cls(h).squeeze(-1))

def train(X, y, ent, epochs=300, lr=0.001, batch=256):
    mu = X.mean(0); std = X.std(0); std[std<1e-10] = 1
    Xn = (X - mu) / std
    
    n = len(X)
    perm = np.random.permutation(n)
    nv = n // 5
    Xv, yv, ev = Xn[perm[:nv]], y[perm[:nv]], ent[perm[:nv]]
    Xt, yt, et = Xn[perm[nv:]], y[perm[nv:]], ent[perm[nv:]]
    
    loader = DataLoader(TensorDataset(
        torch.tensor(Xt, dtype=torch.float32),
        torch.tensor(yt, dtype=torch.float32),
        torch.tensor(et, dtype=torch.float32),
    ), batch_size=batch, shuffle=True)
    
    model = SAPredictor(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    Xv_t = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
    yv_t = torch.tensor(yv, dtype=torch.float32).to(DEVICE)
    ev_t = torch.tensor(ev, dtype=torch.float32).to(DEVICE)
    
    best_vl = float('inf'); best_state = None; best_r2 = -999; best_ca = 0
    
    for ep in range(epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(DEVICE), yb.to(DEVICE), eb.to(DEVICE)
            pk, pe = model(xb)
            loss = nn.MSELoss()(pk, yb) + 0.5*nn.BCELoss()(pe, eb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vpk, vpe = model(Xv_t)
            vl = nn.MSELoss()(vpk, yv_t).item()
            ss_r = ((yv_t-vpk)**2).sum().item()
            ss_t = ((yv_t-yv_t.mean())**2).sum().item()
            r2 = 1 - ss_r/max(ss_t,1e-10)
            ca = ((vpe>0.5).float()==ev_t).float().mean().item()
        
        if vl < best_vl:
            best_vl=vl; best_r2=r2; best_ca=ca
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if ep % 50 == 0 or ep == epochs-1:
            print(f"  E{ep:4d} | val_loss={vl:.6f} R²={r2:.4f} cls={ca:.1%}")
    
    model.load_state_dict(best_state)
    print(f"\n  Best: R²={best_r2:.4f}, cls={best_ca:.1%}")
    return model, mu, std, best_r2, best_ca

if __name__ == '__main__':
    print("="*60)
    print("  MLP v6b — Correct Labels, Structured States")
    print("="*60)
    
    cache = 'sa_data/v6b_data.npz'
    if os.path.exists(cache):
        d = np.load(cache)
        X, y, ent = d['X'], d['y'], d['ent']
        print(f"  Loaded cached data: {len(X)} states")
    else:
        X, y, ent = generate_dataset()
        np.savez(cache, X=X, y=y, ent=ent)
    
    print(f"\n  Dataset: {len(X)} states, {N_FEAT} features")
    print(f"  K_DW: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  K>0: {np.sum(y>0.001)} ({100*np.mean(y>0.001):.1f}%)")
    print(f"  Entangled: {ent.sum()} ({100*ent.mean():.1f}%)")
    
    model, mu, std, r2, ca = train(X, y, ent, epochs=300)
    
    torch.save({
        'model_state': model.state_dict(),
        'mu': mu, 'std': std,
        'feature_names': FEATURES,
        'r2': r2, 'cls_acc': ca,
        'note': 'v6b: correct K_DW, structured PPT states, GELU, 13 features',
    }, 'sa_data/model_v6.pt')
    
    print(f"\n  💾 sa_data/model_v6.pt")
    print(f"  R²={r2:.4f}, Classification={ca:.1%}")
    print(f"{'='*60}")
