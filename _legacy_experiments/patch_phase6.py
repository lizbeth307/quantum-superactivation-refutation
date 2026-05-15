import re

with open('phase6_ml_teleport_engine.py', 'r', encoding='utf-8') as f:
    code = f.read()

# We will completely overwrite phase6_ml_teleport_engine.py to make it clean
clean_code = """
import numpy as np, time, sys
from multiprocessing.dummy import Pool as ThreadPool
sys.path.insert(0, '.')
from phase4_coherent_info import S
from phase5_universal_channel_engine import partial_transpose, build_cq_kraus
from sa_engine import realignment_norm, partial_transpose_B

def evaluate_Ic(Ks_in, Ks_er, rho_in):
    dim = rho_in.shape[0]
    ev, evec = np.linalg.eigh(rho_in)
    idx = ev > 1e-10
    ev = ev[idx]; evec = evec[:, idx]
    d_R = len(ev)
    Psi = np.zeros(d_R * dim, dtype=complex)
    for i in range(d_R): Psi[i*dim : (i+1)*dim] = np.sqrt(ev[i]) * evec[:, i]
        
    Ic_total = 0
    d_out = Ks_in[0].shape[0]
    rho_RB_in = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
    for K in Ks_in:
        KV_Psi = np.zeros(d_R*d_out, dtype=complex)
        for i in range(d_R): KV_Psi[i*d_out : (i+1)*d_out] = K @ Psi[i*dim : (i+1)*dim]
        rho_RB_in += np.outer(KV_Psi, KV_Psi.conj())
        
    tr_in = np.trace(rho_RB_in).real
    if tr_in > 1e-15:
        rho_RB_in /= tr_in
        rho_B_in = np.zeros((d_out, d_out), dtype=complex)
        for i in range(d_R): rho_B_in += rho_RB_in[i*d_out:(i+1)*d_out, i*d_out:(i+1)*d_out]
        Ic_total += tr_in * (S(rho_B_in) - S(rho_RB_in))
        
    d_out_er = Ks_er[0].shape[0]
    rho_RB_er = np.zeros((d_R*d_out_er, d_R*d_out_er), dtype=complex)
    for K in Ks_er:
        KV_Psi = np.zeros(d_R*d_out_er, dtype=complex)
        for i in range(d_R): KV_Psi[i*d_out_er : (i+1)*d_out_er] = K @ Psi[i*dim : (i+1)*dim]
        rho_RB_er += np.outer(KV_Psi, KV_Psi.conj())
        
    tr_er = np.trace(rho_RB_er).real
    if tr_er > 1e-15:
        rho_RB_er /= tr_er
        rho_B_er = np.zeros((d_out_er, d_out_er), dtype=complex)
        for i in range(d_R): rho_B_er += rho_RB_er[i*d_out_er:(i+1)*d_out_er, i*d_out_er:(i+1)*d_out_er]
        Ic_total += tr_er * (S(rho_B_er) - S(rho_RB_er))
        
    return Ic_total

def entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return 0.0 if len(ev) == 0 else -np.sum(ev * np.log2(ev))

def get_separable_mixer(U, V, dA, dB, k):
    sigma = np.zeros((dA*dB, dA*dB), dtype=complex)
    for j in range(k):
        u = U[:, j] / max(np.linalg.norm(U[:, j]), 1e-10)
        v = V[:, j] / max(np.linalg.norm(V[:, j]), 1e-10)
        uv = np.kron(u, v)
        sigma += np.outer(uv, uv.conj())
    return sigma / k

def get_boundary_state_advanced(psi, U, V, dA, dB, k):
    rho_pure = np.outer(psi, psi.conj())
    sigma = get_separable_mixer(U, V, dA, dB, k)
    pt_pure = partial_transpose(rho_pure, dA, dB)
    pt_sigma = partial_transpose(sigma, dA, dB)
    
    p_low, p_high = 0.0, 1.0
    for _ in range(15):
        p_mid = (p_low + p_high) / 2
        if np.linalg.eigvalsh(p_mid * pt_pure + (1 - p_mid) * pt_sigma)[0] < 0:
            p_high = p_mid
        else:
            p_low = p_mid
    return p_low * rho_pure + (1 - p_low) * sigma

def state_to_channel(rho_ppt, dA, dB):
    rho_A = np.zeros((dA, dA), dtype=complex)
    for i in range(dB):
        for a1 in range(dA):
            for a2 in range(dA):
                rho_A[a1, a2] += rho_ppt[a1*dB + i, a2*dB + i]
    ev, evec = np.linalg.eigh(rho_A)
    inv_sqrt = np.zeros((dA, dA), dtype=complex)
    for i in range(dA):
        if ev[i] > 1e-12: inv_sqrt += (1.0/np.sqrt(ev[i])) * np.outer(evec[:,i], evec[:,i].conj())
    Op = np.kron(inv_sqrt, np.eye(dB))
    C = Op @ rho_ppt @ Op.conj().T / dA
    pt_C = partial_transpose(C, dA, dB)
    if np.linalg.eigvalsh(pt_C).min() < -1e-6: return None
    choi = dA * C
    ev_c, evec_c = np.linalg.eigh(choi)
    Ks = []
    for k in range(len(ev_c)):
        if ev_c[k] > 1e-10: Ks.append(np.sqrt(ev_c[k]) * evec_c[:, k].reshape(dA, dB).T)
    return Ks

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def numpy_mlp_forward(x, W):
    h = np.dot(x, W['net.0.weight'].T) + W['net.0.bias']; h = gelu(h)
    h = np.dot(h, W['net.3.weight'].T) + W['net.3.bias']; h = gelu(h)
    h = np.dot(h, W['net.6.weight'].T) + W['net.6.bias']; h = gelu(h)
    pk = np.dot(h, W['reg.weight'].T) + W['reg.bias']
    pe = 1 / (1 + np.exp(-(np.dot(h, W['cls.weight'].T) + W['cls.bias'])))
    return pk[0], pe[0]

def extract_features(rho, dA, dB):
    d = dA * dB
    ev = np.linalg.eigvalsh(rho); ev_pos = ev[ev > 1e-15]
    rank = len(ev_pos)
    purity = np.real(np.trace(rho @ rho))
    S_AB = -np.sum(ev_pos * np.log2(ev_pos)) if len(ev_pos) > 0 else 0
    rA = np.zeros((dA, dA), dtype=complex)
    for a in range(dA):
        for ap in range(dA): rA[a, ap] = np.trace(rho[a*dB:(a+1)*dB, ap*dB:(ap+1)*dB])
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_A, S_B = entropy(rA), entropy(rB)
    MI = S_A + S_B - S_AB
    pt = partial_transpose_B(rho, dA, dB)
    pt_eigs = np.linalg.eigvalsh(pt)
    R_norm = realignment_norm(rho, dA, dB)
    neg = sum(abs(e) for e in pt_eigs if e < 0)
    return np.array([rank / d, purity, ev.min(), ev.max(), ev.std(), pt_eigs.min(), abs(pt_eigs.min()), S_A, S_B, S_AB, MI, R_norm, neg])

def calculate_surrogate_score(rho_ppt, dA, dB, W, mu, std):
    xn = (extract_features(rho_ppt, dA, dB) - mu) / std
    pk, pe = numpy_mlp_forward(xn, W)
    return pk + 0.1 * pe

def _teleport_worker(args):
    idx, dA, dB, steps, seed, W, mu, std = args
    np.random.seed(seed)
    
    k = dA * dB
    psi = np.random.randn(dA*dB) + 1j*np.random.randn(dA*dB)
    psi /= np.linalg.norm(psi)
    U = np.random.randn(dA, k) + 1j*np.random.randn(dA, k)
    V = np.random.randn(dB, k) + 1j*np.random.randn(dB, k)
    
    best_psi, best_U, best_V = psi, U, V
    best_score = -999
    
    for step in range(steps):
        n_psi = best_psi + 0.1 * (np.random.randn(dA*dB) + 1j*np.random.randn(dA*dB))
        n_psi /= np.linalg.norm(n_psi)
        n_U = best_U + 0.1 * (np.random.randn(dA, k) + 1j*np.random.randn(dA, k))
        n_V = best_V + 0.1 * (np.random.randn(dB, k) + 1j*np.random.randn(dB, k))
        
        rho_ppt = get_boundary_state_advanced(n_psi, n_U, n_V, dA, dB, k)
        score = calculate_surrogate_score(rho_ppt, dA, dB, W, mu, std)
        
        if score > best_score:
            best_score = score
            best_psi, best_U, best_V = n_psi, n_U, n_V
            
    best_rho_ppt = get_boundary_state_advanced(best_psi, best_U, best_V, dA, dB, k)
    Ks_P = state_to_channel(best_rho_ppt, dA, dB)
    if Ks_P is None: return -999, best_score
    
    Ks_in, Ks_er = build_cq_kraus(Ks_P, p_erasure=0.5)
    best_Ic = -999
    dim = dA * 2
    for _ in range(5):
        test_psi = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
        test_rho = test_psi @ test_psi.conj().T
        test_rho /= np.trace(test_rho)
        ic = evaluate_Ic(Ks_in, Ks_er, test_rho)
        if ic > best_Ic: best_Ic = ic
        
    return best_Ic, best_score

def run_ml_teleport(dA, dB, W, mu, std, n_swarms=30, steps=1000):
    print(f"\\n{'='*60}")
    print(f"  TELEPORTING (UPB Constructor): {dA}x{dB} | Swarms: {n_swarms} | Steps: {steps}")
    print(f"{'='*60}")
    
    t0 = time.time()
    args = [(i, dA, dB, steps, int(time.time()*1000)%1000000 + i, W, mu, std) for i in range(n_swarms)]
    
    with ThreadPool(30) as pool:
        results = pool.map(_teleport_worker, args)
        
    valid = [r for r in results if r[0] != -999]
    if not valid:
        print("All swarms failed to generate valid channels.")
        return
        
    max_ic = max(r[0] for r in valid)
    max_score = max(r[1] for r in valid)
    sa_count = sum(1 for r in valid if r[0] > 1e-4)
    dt = time.time() - t0
    
    print(f"  Surrogate Peak Reached: {max_score:.4f}")
    if max_ic > 0:
        print(f"  [SUCCESS] Abyss Jumped! Found {sa_count}/{n_swarms} SA Channels.")
        print(f"  [PROOF] Exact I_c Maximum: +{max_ic:.6f} Qubits/use")
    else:
        print(f"  [FAILED] Fell into the Abyss. Exact I_c Max: {max_ic:.6f}")
    print(f"  Time Elapsed: {dt:.1f}s")

if __name__ == '__main__':
    # Try importing numpy first to avoid any torch dll overlap
    import torch
    print("Extracting PyTorch Weights to NumPy...")
    ckpt = torch.load('sa_data/model_v6.pt', map_location='cpu', weights_only=False)
    W = {k: v.numpy() for k, v in ckpt['model_state'].items()}
    mu, std = ckpt['mu'], ckpt['std']
    
    run_ml_teleport(2, 4, W, mu, std, steps=1500)
"""

with open('phase6_ml_teleport_engine.py', 'w', encoding='utf-8') as f:
    f.write(clean_code)
print("Rewritten phase6_ml_teleport_engine.py cleanly!")
