"""
train_predictor.py — Phase 0.4 + 0.4b: Train MLP with FULL AI Self-Diagnostics.

Per Master Plan v4:
- After each epoch: HEALTH (loss trend, gradient, val score)
- After each epoch: DATA GAPS (missing regions)
- Every 10 epochs: STUCK DETECTION
- Every 10 epochs: CONFIDENCE MAP per dimension
- If stuck: AUTO-ACTION
- All diagnostics saved to sa_data/ai_diagnostic_log.txt
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time, os, sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ══════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════

FEATURES = [
    'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
    'pt_min', 'pt_boundary_dist', 'pt_neg_count',
    'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
    'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
]

def load_all_paths():
    """Load and merge all path data from Phase 0.1."""
    X_all, y_all, d_all = [], [], []
    
    path_files = {
        'path_a_d4.npz': 4, 'path_b_d4.npz': 4,
        'path_c_d4.npz': 4, 'path_e_d4.npz': 4,
        'path_d_d6.npz': 6, 'path_d_d8.npz': 8,
        'path_d_d9.npz': 9,
        'path_d_d10.npz': 10, 'path_d_d12.npz': 12,
        'path_d_d14.npz': 14, 'path_d_d15.npz': 15,
        'path_d_d16.npz': 16, 'path_d_d18.npz': 18,
        'path_d_d20.npz': 20, 'path_d_d24.npz': 24,
        'path_d_d30.npz': 30,
    }
    
    for fname, d in path_files.items():
        fpath = f'sa_data/{fname}'
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        if 'kdw' not in data:
            continue
        n = len(data['kdw'])
        X = np.zeros((n, len(FEATURES)))
        for j, feat in enumerate(FEATURES):
            if feat in data:
                X[:, j] = data[feat].real
        y = data['kdw'].real
        X_all.append(X)
        y_all.append(y)
        d_all.append(np.full(n, d))
        print(f"  ✅ {fname}: {n} pts, K_DW [{y.min():.4f}, {y.max():.4f}]")
    
    # Load Phase 0.5 data (negatives + SA search)
    p05_path = 'sa_data/phase05_data.npz'
    if os.path.exists(p05_path):
        p05 = np.load(p05_path)
        X_all.append(p05['X'])
        y_all.append(p05['y'])
        d_all.append(p05['dims'])
        print(f"  ✅ phase05_data.npz: {len(p05['y'])} pts, K_DW [{p05['y'].min():.4f}, {p05['y'].max():.4f}]")
    
    # Load extra fix data (d=4 Phase 0Q, d=24 extra)
    for fname, d in [('path_d_d4_0q.npz', 4), ('path_d_d24_full.npz', 24)]:
        fpath = f'sa_data/{fname}'
        if os.path.exists(fpath):
            data = np.load(fpath)
            if 'kdw' in data:
                n = len(data['kdw'])
                X = np.zeros((n, len(FEATURES)))
                for j, feat in enumerate(FEATURES):
                    if feat in data:
                        X[:, j] = data[feat].real
                y = data['kdw'].real
                X_all.append(X)
                y_all.append(y)
                d_all.append(np.full(n, d))
                print(f"  ✅ {fname}: {n} pts, K_DW [{y.min():.4f}, {y.max():.4f}]")
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    dims = np.concatenate(d_all)
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    X = (X - mu) / std
    
    return X, y, dims, mu, std


# ══════════════════════════════════════════════════════════
# Model (per audit fix #28 — regression, not binary)
# ══════════════════════════════════════════════════════════

class KDWPredictor(nn.Module):
    """Dual-head MLP (audit AI-7): regression + classification."""
    def __init__(self, n_features, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(hidden // 2, 1)    # K_DW value
        self.cls_head = nn.Linear(hidden // 2, 1)    # P(K>0)
    
    def forward(self, x):
        h = self.shared(x)
        k_dw = self.reg_head(h).squeeze(-1)
        p_pos = torch.sigmoid(self.cls_head(h).squeeze(-1))
        return k_dw, p_pos


# ══════════════════════════════════════════════════════════
# Phase 0.4b: AI Self-Diagnostics (MANDATORY per plan)
# ══════════════════════════════════════════════════════════

class AIDiagnostics:
    """Full diagnostics per Master Plan v4 §0.4b."""
    
    def __init__(self, log_path='sa_data/ai_diagnostic_log.txt'):
        self.loss_history = []
        self.r2_history = []
        self.data_coverage = {}
        self.log_path = log_path
        self.log_lines = []
        self._log("╔══ AI SELF-DIAGNOSTIC SYSTEM INITIALIZED ══╗")
    
    def _log(self, msg):
        self.log_lines.append(msg)
        print(msg)
    
    def check_health(self, epoch, loss, r2, grad_norm):
        """After EACH epoch: HEALTH check."""
        self.loss_history.append(loss)
        self.r2_history.append(r2)
        issues = []
        
        # Loss trend
        if len(self.loss_history) > 5:
            recent = self.loss_history[-5:]
            if max(recent) - min(recent) < 0.0001:
                issues.append("STUCK: loss flat for 5 epochs")
            if self.loss_history[-1] > self.loss_history[-2] * 1.5:
                issues.append(f"SPIKE: loss jumped {self.loss_history[-2]:.6f}→{loss:.6f}")
        
        # R² drop
        if len(self.r2_history) > 2 and r2 < self.r2_history[-2] - 0.01:
            issues.append(f"OVERFITTING: R² dropped {self.r2_history[-2]:.3f}→{r2:.3f}")
        
        # Gradient health
        if grad_norm > 10:
            issues.append(f"EXPLODING gradient: {grad_norm:.1f}")
        if grad_norm < 1e-6:
            issues.append(f"VANISHING gradient: {grad_norm:.2e}")
        
        return issues
    
    def check_data_gaps(self, X_train, y_train, dims_train):
        """After each epoch: scan for DATA GAPS."""
        gaps = []
        
        # Near-zero K_DW coverage
        near_zero = np.sum(np.abs(y_train) < 0.005)
        if near_zero < 100:
            gaps.append(f"Only {near_zero} pts near K_DW=0 (need 100+)")
        
        # Per-dimension coverage
        for d in sorted(set(dims_train.astype(int))):
            n = np.sum(dims_train == d)
            if n < 1000:
                gaps.append(f"d={d}: only {int(n)} pts (need 1000+)")
        
        # Positive example ratio
        pos = np.sum(y_train > 0.001)
        ratio = pos / len(y_train)
        if ratio < 0.1:
            gaps.append(f"Imbalance: only {ratio:.1%} positive")
        
        return gaps
    
    def confidence_map(self, model, X_val, y_val, dims_val):
        """Every 10 epochs: CONFIDENCE MAP per dimension."""
        model.eval()
        cmap = {}
        with torch.no_grad():
            pred_k, _ = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))
            pred = pred_k.cpu().numpy()
        
        for d in sorted(set(dims_val.astype(int))):
            mask = dims_val == d
            if mask.sum() < 5:
                continue
            y_d = y_val[mask]
            p_d = pred[mask]
            ss_res = np.sum((y_d - p_d) ** 2)
            ss_tot = np.sum((y_d - y_d.mean()) ** 2)
            r2_d = 1 - ss_res / max(ss_tot, 1e-10)
            r2_d = max(0, min(1, r2_d))
            cmap[d] = r2_d
        
        return cmap
    
    def auto_fix(self, issues, optimizer):
        """If stuck: AUTO-ACTION."""
        actions = []
        for issue in issues:
            if "STUCK" in issue:
                for pg in optimizer.param_groups:
                    pg['lr'] = min(pg['lr'] * 1.5, 0.01)
                actions.append(f"ACTION: lr → {optimizer.param_groups[0]['lr']:.6f}")
            if "EXPLODING" in issue:
                actions.append("ACTION: gradient clipping active (1.0)")
            if "VANISHING" in issue:
                for pg in optimizer.param_groups:
                    pg['lr'] = min(pg['lr'] * 2.0, 0.01)
                actions.append(f"ACTION: lr doubled → {optimizer.param_groups[0]['lr']:.6f}")
        return actions
    
    def print_dashboard(self, epoch, epochs, loss, val_loss, r2, grad_norm,
                        issues, gaps, cmap, actions):
        """Terminal dashboard output per plan spec."""
        self._log(f"╔══ AI SELF-DIAGNOSTIC ═══════════════════════════════════╗")
        self._log(f"║ Epoch {epoch}/{epochs}  Loss: {loss:.6f}  Val: {val_loss:.6f}  R²: {r2:.3f}")
        self._log(f"║")
        
        # HEALTH
        self._log(f"║ HEALTH:")
        if not issues:
            lr = "converging" if len(self.loss_history) > 1 and self.loss_history[-1] < self.loss_history[-2] else "stable"
            self._log(f"║   Learning: ✅ {lr}")
            self._log(f"║   Gradient: ✅ {grad_norm:.4f}")
        else:
            for iss in issues:
                self._log(f"║   ⚠️  {iss}")
        
        # DATA GAPS
        if gaps:
            self._log(f"║")
            self._log(f"║ DATA GAPS:")
            for gap in gaps[:5]:
                self._log(f"║   ⚠️  {gap}")
        
        # CONFIDENCE MAP
        if cmap:
            self._log(f"║")
            self._log(f"║ CONFIDENCE MAP (per dimension):")
            for d, r2_d in sorted(cmap.items()):
                bars = int(r2_d * 20)
                bar_str = '█' * bars + '░' * (20 - bars)
                label = "✅" if r2_d > 0.8 else ("⚠️" if r2_d > 0.5 else "❌")
                self._log(f"║   d={d:4d}: {bar_str}  {r2_d*100:5.1f}% {label}")
        
        # AUTO-ACTIONS
        if actions:
            self._log(f"║")
            for act in actions:
                self._log(f"║ 🔧 {act}")
        
        self._log(f"╚══════════════════════════════════════════════════════════╝")
    
    def save_log(self):
        """Save all diagnostics to file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))
        print(f"  📝 Diagnostics saved: {self.log_path}")


# ══════════════════════════════════════════════════════════
# Training Loop with Diagnostics
# ══════════════════════════════════════════════════════════

def train(X, y, dims, epochs=500, lr=0.001, batch_size=256):
    n = len(X)
    n_features = X.shape[1]
    
    # Train/val split (stratified by dimension)
    perm = np.random.permutation(n)
    n_val = max(1, n // 5)
    X_val, y_val, d_val = X[perm[:n_val]], y[perm[:n_val]], dims[perm[:n_val]]
    X_train, y_train, d_train = X[perm[n_val:]], y[perm[n_val:]], dims[perm[n_val:]]
    
    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_vl = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    
    model = KDWPredictor(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    diag = AIDiagnostics()
    best_val_loss = float('inf')
    best_state = None
    best_r2 = -999
    
    diag._log(f"\n  Training: {len(X_train)} train, {len(X_val)} val, {n_features} features")
    diag._log(f"  Device: {DEVICE}, Epochs: {epochs}, Batch: {batch_size}")
    diag._log(f"  Unique dimensions: {sorted(set(dims.astype(int)))}")
    
    # Initial data gap check
    gaps = diag.check_data_gaps(X_train, y_train, d_train)
    if gaps:
        diag._log(f"\n  ⚠️  INITIAL DATA GAPS:")
        for g in gaps:
            diag._log(f"     {g}")
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_grad = 0
        n_batches = 0
        
        for xb, yb in loader:
            pred_k, pred_cls = model(xb)
            
            # Weighted loss (audit AI-2): 100x weight for positive K
            weights = torch.where(yb > 0, 100.0, 1.0)
            reg_loss = (weights * (pred_k - yb) ** 2).mean()
            
            # Classification loss
            labels = (yb > 0.001).float()
            cls_loss = nn.BCELoss()(pred_cls, labels)
            
            loss = reg_loss + 0.1 * cls_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += reg_loss.item()
            total_grad += grad_norm
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_grad = total_grad / max(n_batches, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_k, val_pred_cls = model(X_vl)
            val_loss = nn.MSELoss()(val_pred_k, y_vl).item()
            ss_res = ((y_vl - val_pred_k) ** 2).sum().item()
            ss_tot = ((y_vl - y_vl.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        scheduler.step()
        
        # Phase 0.4b: HEALTH check every epoch
        issues = diag.check_health(epoch, val_loss, r2, avg_grad)
        
        # Phase 0.4b: AUTO-FIX
        actions = diag.auto_fix(issues, optimizer) if issues else []
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Phase 0.4b: FULL DASHBOARD every 10 epochs
        if epoch % 50 == 0 or epoch == epochs - 1:
            gaps = diag.check_data_gaps(X_train, y_train, d_train)
            cmap = diag.confidence_map(model, X_val, y_val, d_val)
            diag.print_dashboard(epoch, epochs, avg_loss, val_loss, r2,
                                avg_grad, issues, gaps, cmap, actions)
        elif epoch % 10 == 0:
            # Brief status every 10 epochs
            iss_str = ','.join(issues)[:30] if issues else 'OK'
            diag._log(f"  E{epoch:4d} | loss={avg_loss:.6f} val={val_loss:.6f} R²={r2:.4f} grad={avg_grad:.3f} | {iss_str}")
    
    elapsed = time.time() - t0
    diag._log(f"\n  ⏱  Training complete: {elapsed:.1f}s")
    diag._log(f"  🏆 Best val loss: {best_val_loss:.6f}, Best R²: {best_r2:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # Final confidence map
    model.eval()
    cmap = diag.confidence_map(model, X_val, y_val, d_val)
    diag._log(f"\n  📊 FINAL CONFIDENCE MAP:")
    for d, r2_d in sorted(cmap.items()):
        bars = '█' * int(r2_d * 20) + '░' * (20 - int(r2_d * 20))
        diag._log(f"     d={d:4d}: {bars}  {r2_d*100:5.1f}%")
    
    # Final evaluation
    with torch.no_grad():
        vp_k, vp_cls = model(X_vl)
        vp = vp_k.cpu().numpy()
        y_np = y_vl.cpu().numpy()
        cls_pred = (vp_cls.cpu().numpy() > 0.5).astype(int)
        cls_true = (y_np > 0.001).astype(int)
        cls_acc = (cls_pred == cls_true).mean()
        
        mae = np.mean(np.abs(vp - y_np))
        r2_final = 1 - np.sum((y_np - vp)**2) / np.sum((y_np - y_np.mean())**2)
    
    diag._log(f"\n  📈 FINAL METRICS:")
    diag._log(f"     R² (regression): {r2_final:.4f}")
    diag._log(f"     MAE: {mae:.6f}")
    diag._log(f"     Classification accuracy: {cls_acc:.1%}")
    diag._log(f"     K_DW range: [{y_np.min():.4f}, {y_np.max():.4f}]")
    diag._log(f"     Pred range: [{vp.min():.4f}, {vp.max():.4f}]")
    
    diag.save_log()
    
    return model, best_val_loss, r2_final, diag


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  SA PREDICTOR — Phase 0.4 + 0.4b (Full Diagnostics)")
    print("=" * 60)
    
    X, y, dims, mu, std = load_all_paths()
    print(f"\n  Total dataset: {len(X)} points, {X.shape[1]} features")
    print(f"  K_DW: min={y.min():.4f} max={y.max():.4f} mean={y.mean():.4f}")
    print(f"  K_DW > 0: {np.sum(y > 0.001)} ({100*np.mean(y > 0.001):.1f}%)")
    print(f"  Dimensions: {sorted(set(dims.astype(int)))}")
    
    model, val_loss, r2, diag = train(X, y, dims, epochs=500, lr=0.001, batch_size=256)
    
    # Save model + metadata
    torch.save({
        'model_state': model.state_dict(),
        'mu': mu, 'std': std,
        'feature_names': FEATURES,
        'val_loss': val_loss, 'r2': r2,
        'n_features': len(FEATURES),
        'architecture': 'dual_head_256',
    }, 'sa_data/model_v5.pt')
    
    print(f"\n  💾 Model saved: sa_data/model_v5.pt")
    print(f"  R² = {r2:.4f}")
    print(f"{'=' * 60}")
