import numpy as np
import json
from pathlib import Path

with open('runs/sim2f_bench/session_hw/session.json') as f:
    session = json.load(f)

d = np.load('runs/sim2f_bench/session_hw/frames.npz')
sample_order = session['indices']
print(f"Sample order: {sample_order}")
print(f"Settle time: {session['settle_s']} s")

def find_zero_order_centroid(img, search_radius=8):
    flat_idx = np.argmax(img)
    cy, cx = np.unravel_index(flat_idx, img.shape)
    r = search_radius
    y0 = max(0, cy - r)
    y1 = min(img.shape[0], cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(img.shape[1], cx + r + 1)
    patch = img[y0:y1, x0:x1]
    bg = np.median(patch)
    patch_sub = np.clip(patch - bg, 0, None)
    total = patch_sub.sum()
    if total < 1e-6:
        return float(cx), float(cy)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx_c = (xs * patch_sub).sum() / total
    cy_c = (ys * patch_sub).sum() / total
    return float(cx_c), float(cy_c)

results = []
for i, idx in enumerate(sample_order):
    a1 = d[f's{idx}_a1']
    a2 = d[f's{idx}_a2']
    cx1, cy1 = find_zero_order_centroid(a1)
    cx2, cy2 = find_zero_order_centroid(a2)
    dx = cx2 - cx1
    dy = cy2 - cy1
    drift = np.sqrt(dx**2 + dy**2)
    results.append({'sample_idx': idx, 'session_order': i,
        'a1_cx': cx1, 'a1_cy': cy1, 'a2_cx': cx2, 'a2_cy': cy2,
        'dx': dx, 'dy': dy, 'drift_px': drift})
    print(f"  sample {idx:2d} (order {i:2d}): A1=({cx1:7.1f},{cy1:7.1f}) A2=({cx2:7.1f},{cy2:7.1f}) drift=({dx:+6.2f},{dy:+6.2f}) mag={drift:6.2f}")

order = np.array([r['session_order'] for r in results])
drifts = np.array([r['drift_px'] for r in results])
dxs = np.array([r['dx'] for r in results])
dys = np.array([r['dy'] for r in results])
a1_cx = np.array([r['a1_cx'] for r in results])
a1_cy = np.array([r['a1_cy'] for r in results])
a2_cx = np.array([r['a2_cx'] for r in results])
a2_cy = np.array([r['a2_cy'] for r in results])

print(f"\n=== Statistics ===")
print(f"Drift magnitude: mean={drifts.mean():.2f}, max={drifts.max():.2f}, std={drifts.std():.2f}")
print(f"Drift dx: mean={dxs.mean():.2f}, std={dxs.std():.2f}")
print(f"Drift dy: mean={dys.mean():.2f}, std={dys.std():.2f}")
print(f"A1 centroid spread: cx std={a1_cx.std():.2f}, cy std={a1_cy.std():.2f}")
print(f"Session 0-order spread (A1): cx range={a1_cx.max()-a1_cx.min():.2f}, cy range={a1_cy.max()-a1_cy.min():.2f}")

print("\n=== Fitting drift vs session order ===")
coeffs_x = np.polyfit(order, dxs, 1)
coeffs_y = np.polyfit(order, dys, 1)
print(f"Linear fit dx vs order: slope={coeffs_x[0]:.4f} px/sample, intercept={coeffs_x[1]:.4f}")
print(f"Linear fit dy vs order: slope={coeffs_y[0]:.4f} px/sample, intercept={coeffs_y[1]:.4f}")

coeffs_x2 = np.polyfit(order, dxs, 2)
coeffs_y2 = np.polyfit(order, dys, 2)
resid_x2 = dxs - np.polyval(coeffs_x2, order)
resid_y2 = dys - np.polyval(coeffs_y2, order)
print(f"Poly2 fit dx: residual std={resid_x2.std():.2f}")
print(f"Poly2 fit dy: residual std={resid_y2.std():.2f}")

coeffs_cx = np.polyfit(order, a1_cx, 1)
coeffs_cy = np.polyfit(order, a1_cy, 1)
print(f"\nSession-level A1 centroid vs order:")
print(f"  cx: slope={coeffs_cx[0]:.4f} px/sample, intercept={coeffs_cx[1]:.2f}")
print(f"  cy: slope={coeffs_cy[0]:.4f} px/sample, intercept={coeffs_cy[1]:.2f}")
resid_cx = a1_cx - np.polyval(coeffs_cx, order)
resid_cy = a1_cy - np.polyval(coeffs_cy, order)
print(f"  cx residual std={resid_cx.std():.2f}, max={np.abs(resid_cx).max():.2f}")
print(f"  cy residual std={resid_cy.std():.2f}, max={np.abs(resid_cy).max():.2f}")

from scipy.optimize import curve_fit
def exp_model(t, a, b, c):
    return a * np.exp(b * t) + c
try:
    popt_x, _ = curve_fit(exp_model, order, a1_cx, p0=[1, 0.1, a1_cx.mean()], maxfev=10000)
    popt_y, _ = curve_fit(exp_model, order, a1_cy, p0=[1, 0.1, a1_cy.mean()], maxfev=10000)
    resid_x_exp = a1_cx - exp_model(order, *popt_x)
    resid_y_exp = a1_cy - exp_model(order, *popt_y)
    print(f"\nExponential fit A1 cx: a={popt_x[0]:.2f}, b={popt_x[1]:.4f}, c={popt_x[2]:.2f}, resid_std={resid_x_exp.std():.2f}")
    print(f"Exponential fit A1 cy: a={popt_y[0]:.2f}, b={popt_y[1]:.4f}, c={popt_y[2]:.2f}, resid_std={resid_y_exp.std():.2f}")
except Exception as e:
    print(f"Exponential fit failed: {e}")

def sin_model(t, a, b, c, d):
    return a * np.sin(b * t + c) + d
try:
    popt_x, _ = curve_fit(sin_model, order, a1_cx, p0=[10, 0.3, 0, a1_cx.mean()], maxfev=10000)
    popt_y, _ = curve_fit(sin_model, order, a1_cy, p0=[10, 0.3, 0, a1_cy.mean()], maxfev=10000)
    resid_x_sin = a1_cx - sin_model(order, *popt_x)
    resid_y_sin = a1_cy - sin_model(order, *popt_y)
    print(f"\nSinusoidal fit A1 cx: amp={popt_x[0]:.2f}, freq={popt_x[1]:.4f}, phase={popt_x[2]:.2f}, offset={popt_x[3]:.2f}, resid_std={resid_x_sin.std():.2f}")
    print(f"Sinusoidal fit A1 cy: amp={popt_y[0]:.2f}, freq={popt_y[1]:.4f}, phase={popt_y[2]:.2f}, offset={popt_y[3]:.2f}, resid_std={resid_y_sin.std():.2f}")
except Exception as e:
    print(f"Sinusoidal fit failed: {e}")

np.savez('runs/sim2f_bench/session_hw/drift_analysis.npz',
         order=order, drifts=drifts, dxs=dxs, dys=dys,
         a1_cx=a1_cx, a1_cy=a1_cy, a2_cx=a2_cx, a2_cy=a2_cy,
         sample_indices=np.array(sample_order))
print("\nSaved drift_analysis.npz")
