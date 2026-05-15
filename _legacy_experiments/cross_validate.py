"""
cross_validate.py — Cross-validate PySR formula and MLP on extreme dimensions.
Tests out-of-distribution generalization for d=40,50,80,100,200.
"""
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    d = dA * dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]; phi = eigvecs[:, mask]; r = len(lam)
    sqrt_lam = np.sqrt(lam)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial == 0: U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA,dA)+1j*np.random.randn(dA,dA)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dA); S_B_x = np.zeros(dA); S_E_x = np.zeros(dA)
        for x in range(dA):
            beta = np.zeros((r, dB), dtype=complex)
            for k in range(r):
                for a in range(dA):
                    beta[k] += U[a,x].conj()*phi[a*dB:(a+1)*dB, k]
            norms_sq = np.array([np.dot(beta[k].conj(), beta[k]).real for k in range(r)])
            p_x[x] = np.dot(lam, norms_sq)
            if p_x[x] < 1e-15: continue
            rho_B_x = sum(sqrt_lam[k]*sqrt_lam[l]*np.outer(beta[k],beta[l].conj()) for k in range(r) for l in range(r))/p_x[x]
            S_B_x[x] = von_neumann(rho_B_x)
            rho_E_x = np.zeros((r,r), dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_E_x[k,l] = sqrt_lam[k]*sqrt_lam[l]*np.dot(beta[l].conj(), beta[k])
            rho_E_x /= p_x[x]
            S_E_x[x] = von_neumann(rho_E_x)
        I_XB = S_B_unc - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        I_XE = S_E_unc - sum(p_x[x]*S_E_x[x] for x in range(dA) if p_x[x]>1e-15)
        best = max(best, I_XB - I_XE)
    return best

def compute_features(rho, dA, dB):
    d = dA * dB
    eigs = np.sort(np.linalg.eigvalsh(rho))
    rA = np.zeros((dA,dA), dtype=complex)
    for a1 in range(dA):
        for a2 in range(dA):
            for b in range(dB):
                rA[a1,a2] += rho[a1*dB+b, a2*dB+b]
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_A = von_neumann(rA); S_B = von_neumann(rB); S_AB = von_neumann(rho)
    return {
        'rank_norm': np.sum(eigs>1e-10)/d, 'purity_norm': np.trace(rho@rho).real*d,
        'eig_min': eigs[0], 'eig_max': eigs[-1], 'eig_std': np.std(eigs),
        'pt_min': 0.0, 'pt_boundary_dist': 0.0, 'pt_neg_count': 0,
        'S_A': S_A, 'S_B': S_B, 'S_AB': S_AB,
        'mutual_info': S_A+S_B-S_AB, 'mutual_info_norm': (S_A+S_B-S_AB)/(2*np.log2(d)),
        'realign_norm': 1.0, 'A_max_mixed_dist': np.linalg.norm(rA-np.eye(dA)/dA),
        'B_max_mixed_dist': np.linalg.norm(rB-np.eye(dB)/dB),
    }

def worker(args):
    rho0, dA, dB, eps, seed = args
    rng = np.random.RandomState(seed)
    d = dA*dB
    noise = rng.randn(d,d)+1j*rng.randn(d,d)
    noise = noise@noise.conj().T; noise /= np.trace(noise)
    rho = (1-eps)*rho0 + eps*noise
    rho = (rho+rho.conj().T)/2; rho /= np.trace(rho).real
    
    feats = compute_features(rho, dA, dB)
    kdw = kdw_stinespring(rho, dA, dB, 20)
    
    pysr_pred = feats['S_B'] + 2.73 * feats['eig_std'] - feats['S_A']
    scaling_pred = 0.950 * np.log2(d) - 1.884
    
    return kdw, pysr_pred, scaling_pred, feats

if __name__ == '__main__':
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']
    
    print("=" * 65)
    print("  CROSS-VALIDATION: PySR vs Scaling Law vs MLP on d=40..200")
    print("=" * 65)
    
    # Load MLP v4
    DEVICE = 'cpu'
    features_list = ['rank_norm','purity_norm','eig_min','eig_max','eig_std','pt_min',
                     'pt_boundary_dist','pt_neg_count','S_A','S_B','S_AB','mutual_info',
                     'mutual_info_norm','realign_norm','A_max_mixed_dist','B_max_mixed_dist']
    info = torch.load('sa_data/model_v4.pt', map_location='cpu', weights_only=False)
    mu = info['mu']; std = info['std']
    mlp = nn.Sequential(nn.Linear(16,512),nn.ReLU(),nn.Dropout(0.1),
                        nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),
                        nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    mlp.load_state_dict(info['model_state']); mlp.eval()
    
    configs = [
        (2, 20, np.kron(rho8, np.eye(5)/5), 40),
        (2, 25, np.kron(rho10, np.eye(5)/5), 50),
        (2, 40, np.kron(rho8, np.eye(10)/10), 80),
        (2, 50, np.kron(rho10, np.eye(10)/10), 100),
    ]
    
    print(f"\n  {'d':>5} {'N':>5} {'K_DW':>10} {'PySR':>10} {'Scale':>10} {'MLP':>10} {'PySR_err':>10} {'Scale_err':>10}")
    print(f"  {'-'*70}")
    
    for dA, dB, rho0, d in configs:
        N_pts = 100
        eps_vals = np.linspace(0.0, 0.3, N_pts)
        args = [(rho0, dA, dB, eps, i) for i, eps in enumerate(eps_vals)]
        
        t0 = time.time()
        with Pool(30) as pool:
            results = pool.map(worker, args)
        elapsed = time.time() - t0
        
        kdw_true = np.array([r[0] for r in results])
        pysr_pred = np.array([r[1] for r in results])
        scale_pred = np.array([r[2] for r in results])
        
        # MLP prediction
        feats_all = [r[3] for r in results]
        X = np.zeros((N_pts, 16))
        for i, f in enumerate(feats_all):
            for j, feat in enumerate(features_list):
                X[i, j] = f.get(feat, 0.0)
        X_norm = (X - mu) / std
        with torch.no_grad():
            mlp_pred = mlp(torch.tensor(X_norm, dtype=torch.float32)).squeeze().numpy()
        
        mae_pysr = np.mean(np.abs(kdw_true - pysr_pred))
        mae_scale = np.mean(np.abs(kdw_true - scale_pred))
        mae_mlp = np.mean(np.abs(kdw_true - mlp_pred))
        
        print(f"  {d:5d} {N_pts:5d} {kdw_true.mean():10.4f} {pysr_pred.mean():10.4f} "
              f"{scale_pred[0]:10.4f} {mlp_pred.mean():10.4f} {mae_pysr:10.4f} {mae_scale:10.4f}")
    
    print(f"\n  Legend: K_DW=true, PySR=S(B)+2.73σ-S(A), Scale=0.95log₂(d)-1.88, MLP=v4")
    print(f"{'='*65}")
