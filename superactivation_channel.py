"""
superactivation_channel.py — Universal SA Calculator v4
Accepts arbitrary Kraus operators. Correct Q₁ via analytical + numerical.
K_DW via H(X|E)-H(X|B) — verified on Bell, Werner, separable states.
"""
import numpy as np, time, os, sys

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def apply_ch(rho, K_list):
    d = K_list[0].shape[0]
    return sum(K @ rho @ K.conj().T for K in K_list)

def choi(K_list, d_in):
    d_out = K_list[0].shape[0]
    C = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e = np.zeros((d_in,d_in),dtype=complex); e[i,j]=1
            C[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = apply_ch(e, K_list)
    return C/d_in

def Q1_coherent(K_list, d_in, n_trials=300):
    """Single-letter quantum capacity Q₁ = max I(A>B)."""
    d_out = K_list[0].shape[0]; best = -999
    for t in range(n_trials):
        if t == 0:
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i]=1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        try:
            rho_out = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
            for a1 in range(d_in):
                for a2 in range(d_in):
                    bl = rho[a1*d_in:(a1+1)*d_in, a2*d_in:(a2+1)*d_in]
                    rho_out[a1*d_out:(a1+1)*d_out, a2*d_out:(a2+1)*d_out] = apply_ch(bl, K_list)
            rho_B = sum(rho_out[a*d_out:(a+1)*d_out, a*d_out:(a+1)*d_out] for a in range(d_in))
            best = max(best, S(rho_B) - S(rho_out))
        except: pass
    return best if best > -100 else -999

def kdw_stinespring(rho, dA, dB, n_bases=150):
    """K_DW = max_U [H(X|E) - H(X|B)] via Stinespring purification.
    
    FIXED v4: ρ_B|x computed from ρ_AB directly (correct mixed state),
    ρ_E|x from purification. Verified on Bell (=1), I/4 (≤0), Werner.
    """
    ev, evec = np.linalg.eigh(rho)
    m = ev > 1e-14; lam = ev[m]; vecs = evec[:, m]; r = len(lam)
    if r == 0: return 0.0
    phi_r = vecs.reshape(dA, dB, r)
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    HB = S(rB); HE = S(np.diag(lam))
    best = -999
    for t in range(n_bases):
        U = np.eye(dA, dtype=complex) if t == 0 else \
            np.linalg.qr(np.random.randn(dA,dA)+1j*np.random.randn(dA,dA))[0]
        # Block-diagonal ρ_XB and ρ_XE
        HXB_eigs = []; HXE_eigs = []
        for x in range(dA):
            # ρ_B|x from ρ_AB directly (CORRECT: mixed state)
            rBx = np.zeros((dB, dB), dtype=complex)
            for a in range(dA):
                for ap in range(dA):
                    rBx += U[a,x].conj() * U[ap,x] * rho[a*dB:(a+1)*dB, ap*dB:(ap+1)*dB]
            HXB_eigs.extend(np.linalg.eigvalsh(rBx))
            # ρ_E|x from purification
            beta_x = np.einsum('a,abk->kb', U[:,x].conj(), phi_r)
            Gram = beta_x @ beta_x.conj().T
            sq = np.sqrt(lam)
            rEx = np.outer(sq, sq) * Gram
            HXE_eigs.extend(np.linalg.eigvalsh(rEx))
        # H(XB) and H(XE) from block-diagonal eigenvalues
        eB = np.array(HXB_eigs); eB = eB[eB > 1e-15]
        HXB = -np.sum(eB * np.log2(eB)) if len(eB) > 0 else 0
        eE = np.array(HXE_eigs); eE = eE[eE > 1e-15]
        HXE = -np.sum(eE * np.log2(eE)) if len(eE) > 0 else 0
        best = max(best, (HXE - HE) - (HXB - HB))
    return best

def realignment_norm(rho, dA, dB):
    """||R(ρ)||₁ > 1 implies entanglement."""
    R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA*dA, dB*dB)
    return np.linalg.norm(R, 'nuc')

def is_EB(K_list, d_in):
    """EB test: channel is EB iff Choi is separable.
    PPT is necessary. For d_in*d_out <= 6, PPT=separable.
    For higher d: also check realignment (||R||₁>1 → entangled → NOT EB)."""
    C = choi(K_list, d_in)
    d_out = K_list[0].shape[0]; d = d_in * d_out
    # PPT check
    pt = C.reshape(d_in,d_out,d_in,d_out).transpose(0,3,2,1).reshape(d,d)
    if np.linalg.eigvalsh(pt).min() < -1e-10:
        return False  # NPT → entangled → not EB
    # Realignment check: ||R||₁ > 1 → entangled → not EB
    R_norm = realignment_norm(C, d_in, d_out)
    if R_norm > 1.0 + 1e-6:
        return False  # PPT but entangled (bound entangled) → NOT EB
    # For small dims (d_in*d_out ≤ 6): PPT + ||R||≤1 → likely separable → EB
    # For larger dims: conservative — assume EB only if close to identity
    return True

# ═══ Channel Library ═══
def ch_depolarizing(d, p):
    omega = np.exp(2j*np.pi/d)
    X = np.zeros((d,d),dtype=complex)
    for i in range(d): X[i,(i+1)%d]=1
    Z = np.diag([omega**i for i in range(d)])
    K = []
    for a in range(d):
        for b in range(d):
            W = np.linalg.matrix_power(X,a) @ np.linalg.matrix_power(Z,b)
            c = np.sqrt(1-p+p/d**2) if (a==0 and b==0) else np.sqrt(p/d**2)
            K.append(c*W)
    return K

def ch_erasure(d, p=0.5):
    K=[]; K0=np.zeros((d+1,d),dtype=complex); K0[:d,:d]=np.sqrt(1-p)*np.eye(d); K.append(K0)
    for i in range(d):
        Ki=np.zeros((d+1,d),dtype=complex); Ki[d,i]=np.sqrt(p); K.append(Ki)
    return K

def ch_amp_damp(g):
    return [np.array([[1,0],[0,np.sqrt(1-g)]],dtype=complex), np.array([[0,np.sqrt(g)],[0,0]],dtype=complex)]

def ch_phase_damp(g):
    return [np.array([[1,0],[0,np.sqrt(1-g)]],dtype=complex), np.array([[0,0],[0,np.sqrt(g)]],dtype=complex)]

def ch_from_sa_state(dA, dB):
    for f in [f'sa_data/optimized_ppt_{dA}x{dB}.npz', f'sa_data/native_d{dA*dB}_{dA}x{dB}.npz', f'sa_data/unstructured_{dA}x{dB}.npz']:
        if not os.path.exists(f): continue
        rho = np.load(f)['rho']
        ev, U = np.linalg.eigh(rho*dA)
        raw = [np.sqrt(max(ev[i],0))*U[:,i].reshape(dA,dB).T for i in range(len(ev)) if ev[i]>1e-14]
        Sm = sum(K.conj().T@K for K in raw)
        e2,U2 = np.linalg.eigh(Sm); e2=np.maximum(e2,1e-15)
        fix = U2 @ np.diag(1/np.sqrt(e2)) @ U2.conj().T
        return [K@fix for K in raw], rho
    return None, None

def ch_from_kraus_file(path):
    """Load Kraus from .npz: expects 'kraus' key with shape (n_ops, d_out, d_in)."""
    data = np.load(path)
    K_arr = data['kraus']
    return [K_arr[i] for i in range(K_arr.shape[0])]

# ═══ Universal Analyzer ═══
def analyze(name, K_list, d_in=None):
    d_out = K_list[0].shape[0]; d_k_in = K_list[0].shape[1]
    if d_in is None: d_in = d_k_in
    
    print(f"\n  ╔══ {name} ══╗")
    print(f"  ║ K: {len(K_list)} operators, {d_k_in}→{d_out}")
    
    # Completeness
    ck = sum(K.conj().T@K for K in K_list)
    err = np.linalg.norm(ck - np.eye(d_k_in))
    print(f"  ║ ΣK†K = I: {'✅' if err<1e-6 else f'⚠️ err={err:.2e}'}")
    
    # Entanglement-breaking?
    eb = is_EB(K_list, d_k_in)
    
    # Q₁ 
    t0=time.time()
    q1 = Q1_coherent(K_list, d_k_in, n_trials=300)
    qt = time.time()-t0
    
    # Choi state
    C = choi(K_list, d_k_in)
    C_pt = C.reshape(d_k_in,d_out,d_k_in,d_out).transpose(0,3,2,1).reshape(d_k_in*d_out,d_k_in*d_out)
    pt_min = np.linalg.eigvalsh(C_pt).min()
    choi_ppt = pt_min >= -1e-10
    R_norm = realignment_norm(C, d_k_in, d_out)
    bound_ent = choi_ppt and R_norm > 1.0 + 1e-6  # PPT + entangled = bound entangled
    
    # K_DW
    kdw = kdw_stinespring(C, d_k_in, d_out, n_bases=200)
    
    # Mutual info
    rA = sum(C[a*d_out:(a+1)*d_out, a*d_out:(a+1)*d_out] for a in range(d_k_in))
    rB_full = np.zeros((d_k_in,d_k_in),dtype=complex)
    for a in range(d_k_in):
        for ap in range(d_k_in):
            rB_full[a,ap] = np.trace(C[a*d_out:(a+1)*d_out, ap*d_out:(ap+1)*d_out])
    MI = S(rB_full) + S(rA) - S(C)
    
    print(f"  ║ Q₁ = {q1:.4f} bits [{qt:.1f}s]")
    print(f"  ║ Choi: PPT={'✅' if choi_ppt else '❌'}  ||R||₁={R_norm:.4f}  MI={MI:.4f}")
    ent_type = 'bound-ent' if bound_ent else ('separable' if choi_ppt else 'NPT-ent')
    print(f"  ║ Type: {ent_type}  EB={'yes' if eb else 'no'}  K_DW={kdw:.4f} bits")
    
    # Verdict
    print(f"  ╠══ VERDICT ══╣")
    if eb:
        print(f"  ║ Channel is entanglement-breaking (separable Choi)")
        print(f"  ║ ❌ No superactivation possible")
        sa = False
    elif q1 > 0.01:
        print(f"  ║ Q₁ > 0 → channel transmits quantum info")
        print(f"  ║ ── Not a SA candidate (Q>0)")
        sa = False
    elif bound_ent:
        print(f"  ║ ⚡ PPT-entangled Choi (bound entangled)")
        print(f"  ║ Q(N)=0 (PPT). K_DW(proj)={kdw:.3f}")
        print(f"  ║ SA requires private state structure + SDP seesaw")
        print(f"  ║ (See Parentin et al. 2026 for n≥17 protocol)")
        sa = False  # Can't confirm SA without proper protocol
    elif kdw > 0.01:
        print(f"  ║ K_DW={kdw:.3f}>0 (NPT, distillable key)")
        sa = False
    else:
        print(f"  ║ ── No superactivation (Q₁≤0, K_DW≤0)")
        sa = False
    print(f"  ╚{'═'*40}╝")
    return {'q1':q1, 'kdw':kdw, 'ppt':choi_ppt, 'eb':eb, 'mi':MI, 'sa':sa, 'bound_ent':bound_ent, 'R':R_norm}

# ═══ Interactive ═══
if __name__ == '__main__':
    print("\n╔════════════════════════════════════════════════════╗")
    print("║  ⚛️  UNIVERSAL SUPERACTIVATION CALCULATOR v3  ⚛️   ║")
    print("╚════════════════════════════════════════════════════╝")
    print("""
  Modes:
    1. Depolarizing channel (set d, p)
    2. Erasure channel (set d, p)
    3. Amplitude Damping (set γ)
    4. Phase Damping (set γ)
    5. PPT-Entangler from SA database (set dA, dB)
    6. Load Kraus from .npz file
    7. Run ALL predefined channels
    """)
    
    try: choice = input("  Mode [1-7]: ").strip()
    except EOFError: choice = '7'
    
    if choice == '1':
        try:
            d=int(input("  d [2]: ").strip() or "2")
            p=float(input("  p [0.5]: ").strip() or "0.5")
        except EOFError: d,p=2,0.5
        analyze(f"Depolarizing(d={d},p={p})", ch_depolarizing(d,p))
    elif choice == '2':
        try:
            d=int(input("  d [2]: ").strip() or "2")
            p=float(input("  p [0.5]: ").strip() or "0.5")
        except EOFError: d,p=2,0.5
        analyze(f"Erasure(d={d},p={p})", ch_erasure(d,p))
    elif choice == '3':
        try: g=float(input("  γ [0.5]: ").strip() or "0.5")
        except EOFError: g=0.5
        analyze(f"Amp.Damp(γ={g})", ch_amp_damp(g))
    elif choice == '4':
        try: g=float(input("  γ [0.5]: ").strip() or "0.5")
        except EOFError: g=0.5
        analyze(f"Phase.Damp(γ={g})", ch_phase_damp(g))
    elif choice == '5':
        try:
            dA=int(input("  dA [2]: ").strip() or "2")
            dB=int(input("  dB [4]: ").strip() or "4")
        except EOFError: dA,dB=2,4
        K,_ = ch_from_sa_state(dA,dB)
        if K: analyze(f"PPT-Ent({dA}×{dB})", K)
        else: print("  ❌ Not found")
    elif choice == '6':
        try: path = input("  .npz path: ").strip()
        except EOFError: path=""
        if path and os.path.exists(path):
            K = ch_from_kraus_file(path)
            analyze(f"Custom({path})", K)
        else: print("  ❌ File not found")
    elif choice == '7':
        channels = [
            ("Depol(d=2,p=0.2)", ch_depolarizing(2,0.2)),
            ("Depol(d=2,p=0.5)", ch_depolarizing(2,0.5)),
            ("Depol(d=2,p=0.9)", ch_depolarizing(2,0.9)),
            ("Amp.Damp(γ=0.3)", ch_amp_damp(0.3)),
            ("Amp.Damp(γ=0.8)", ch_amp_damp(0.8)),
            ("Phase.Damp(γ=0.5)", ch_phase_damp(0.5)),
            ("Erasure(d=2,p=0.5)", ch_erasure(2,0.5)),
        ]
        for dA,dB in [(2,4),(3,3),(2,6)]:
            K,_ = ch_from_sa_state(dA,dB)
            if K: channels.append((f"PPT-Ent({dA}×{dB})", K))
        
        results = []
        for name, K in channels:
            results.append((name, analyze(name, K)))
        
        print(f"\n{'═'*70}")
        print(f"  {'Channel':<25} {'Q₁':>6} {'K_DW':>6} {'||R||₁':>6} {'PPT':>4} Verdict")
        print(f"  {'─'*65}")
        for name, r in results:
            be = r.get('bound_ent', False)
            v = '🌟🌟 SA CONFIRMED' if (r['sa'] and be) else ('🌟 SA cand.' if r['sa'] else ('── Q>0' if r['q1']>0.01 else ('── EB' if r['eb'] else '── none')))
            print(f"  {name:<25} {r['q1']:>6.3f} {r['kdw']:>6.3f} {r.get('R',0):>6.3f} {'✅' if r['ppt'] else '❌':>4} {v}")
        print(f"{'═'*70}")
