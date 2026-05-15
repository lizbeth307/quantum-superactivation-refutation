"""
sa_WHY_v2.py — Deep investigation using VERIFIED states from sa_data/.

Uses our 21+ verified SA states to understand the PHYSICS.
"""
import numpy as np, sys, glob, os
sys.path.insert(0, '.')
from sa_engine import S, partial_trace_A, partial_trace_B, partial_transpose_B, kdw_correct, realignment_norm

def load_state(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    rho = data['rho'] if 'rho' in keys else data[keys[0]]
    return rho

print("="*70)
print("  WHY DOES SA WORK? — Analysis on VERIFIED States")
print("="*70)

# Load all native states
states = []
for f in sorted(glob.glob('sa_data/native_d*.npz')):
    name = os.path.basename(f).replace('.npz','').replace('native_','')
    parts = name.split('_')
    d_str = parts[0]  # e.g. 'd12'
    split_str = parts[1]  # e.g. '2x6'
    dA, dB = map(int, split_str.split('x'))
    rho = load_state(f)
    states.append({'name': name, 'dA': dA, 'dB': dB, 'rho': rho, 'file': f})

# Add optimized states
for f in sorted(glob.glob('sa_data/optimized_ppt_*.npz')):
    name = os.path.basename(f).replace('.npz','').replace('optimized_ppt_','')
    dA, dB = map(int, name.split('x'))
    rho = load_state(f)
    states.append({'name': f'opt_{dA}x{dB}', 'dA': dA, 'dB': dB, 'rho': rho, 'file': f})

print(f"\n  Loaded {len(states)} verified states\n")

# Compute properties for ALL states
results = []
for s in states:
    dA, dB = s['dA'], s['dB']
    rho = s['rho']
    d = dA * dB
    
    rho_A = partial_trace_B(rho, dA, dB)
    rho_B = partial_trace_A(rho, dA, dB)
    pt = partial_transpose_B(rho, dA, dB)
    
    sa_ent = S(rho_A)
    sb_ent = S(rho_B)
    sab = S(rho)
    mi = sa_ent + sb_ent - sab
    
    evals = np.linalg.eigvalsh(rho)
    sigma_eig = np.std(evals[evals > 1e-15])
    rank = np.sum(evals > 1e-10)
    
    pt_min = np.linalg.eigvalsh(pt).min()
    R = realignment_norm(rho, dA, dB)
    
    kdw = kdw_correct(rho, dA, dB, 300)
    
    results.append({
        **s, 'S_A': sa_ent, 'S_B': sb_ent, 'S_AB': sab, 'MI': mi,
        'sigma_eig': sigma_eig, 'rank': rank, 'pt_min': pt_min, 'R': R, 'K_DW': kdw,
        'S_B_ratio': sb_ent / np.log2(dB) if dB > 1 else 0,
        'asymmetry': sb_ent - sa_ent,
        'cond_ent': sab - sb_ent,  # S(A|B)
    })

# ═══ Print full table ═══
print(f"  {'Name':>16} {'dA':>3} {'dB':>3} {'K_DW':>7} {'S(A)':>6} {'S(B)':>6} {'S(B)/max':>8} {'S(B)-S(A)':>9} {'σ_eig':>7} {'λ_PT':>8} {'R':>6}")
print(f"  {'-'*95}")
for r in sorted(results, key=lambda x: x['K_DW'], reverse=True):
    print(f"  {r['name']:>16} {r['dA']:3d} {r['dB']:3d} {r['K_DW']:+7.3f} {r['S_A']:6.3f} {r['S_B']:6.3f} {r['S_B_ratio']:8.3f} {r['asymmetry']:+9.3f} {r['sigma_eig']:7.4f} {r['pt_min']:+8.5f} {r['R']:6.3f}")

# ═══ Q1: WHY dA=2 optimal? ═══
print(f"\n{'='*70}")
print("  Q1 ANSWER: WHY is dA=2 optimal?")
print("="*70)
# Group by same d, different splits
d_groups = {}
for r in results:
    d = r['dA'] * r['dB']
    if d not in d_groups: d_groups[d] = []
    d_groups[d].append(r)

for d in sorted(d_groups.keys()):
    if len(d_groups[d]) > 1:
        print(f"\n  d={d}:")
        for r in sorted(d_groups[d], key=lambda x: x['K_DW'], reverse=True):
            print(f"    {r['dA']}×{r['dB']}: K_DW={r['K_DW']:+.3f}, S(A|B)={r['cond_ent']:.3f}, S(B)-S(A)={r['asymmetry']:+.3f}")
        best = max(d_groups[d], key=lambda x: x['K_DW'])
        print(f"    → Best: {best['dA']}×{best['dB']} (dA={best['dA']})")

# ═══ Q2: WHY K_DW ~ log₂(dB)? ═══
print(f"\n{'='*70}")
print("  Q2 ANSWER: WHY K_DW ~ log₂(dB)?")
print("="*70)
dA2 = [r for r in results if r['dA'] == 2]
if dA2:
    print(f"\n  States with dA=2:")
    for r in sorted(dA2, key=lambda x: x['dB']):
        ub = np.log2(r['dB'])
        print(f"    dB={r['dB']:2d}: K_DW={r['K_DW']:+.3f}, S(B)={r['S_B']:.3f}, log₂(dB)={ub:.3f}, K/S(B)={r['K_DW']/max(r['S_B'],1e-10):.3f}, S(B)/max={r['S_B_ratio']:.3f}")

# ═══ Q3: Correlation analysis ═══
print(f"\n{'='*70}")
print("  CORRELATION ANALYSIS")
print("="*70)
kdws = np.array([r['K_DW'] for r in results])
sbs = np.array([r['S_B'] for r in results])
sas = np.array([r['S_A'] for r in results])
sigs = np.array([r['sigma_eig'] for r in results])
asym = np.array([r['asymmetry'] for r in results])
ptmins = np.array([r['pt_min'] for r in results])

for name, arr in [('S(B)', sbs), ('S(A)', sas), ('S(B)-S(A)', asym), 
                   ('σ_eig', sigs), ('λ_min(PT)', ptmins)]:
    if len(arr) > 2 and np.std(arr) > 1e-10:
        corr = np.corrcoef(kdws, arr)[0,1]
        print(f"  corr(K_DW, {name:>10}) = {corr:+.4f}")

# ═══ Test PySR formula ═══
print(f"\n  PySR formula test: K_DW ≈ S(B) + 2.73·σ_eig - S(A)")
predictions = sbs + 2.73*sigs - sas
residuals = kdws - predictions
rmse = np.sqrt(np.mean(residuals**2))
r2 = 1 - np.sum(residuals**2)/np.sum((kdws - kdws.mean())**2)
print(f"  RMSE = {rmse:.4f}, R² = {r2:.4f}")

print(f"\n{'='*70}")
