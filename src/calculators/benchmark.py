import numpy as np
import time


class SGD_Py:
    def __init__(self, dim, lr=1.0):
        self.dim = dim
        self.lr = lr
        self.t = 0

    def update(self, grad):
        self.t += 1
        return self.lr * grad


class Adam_Py:
    def __init__(self, dim, lr=1.0, beta1=0.9, beta2=0.999):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class AdamW_Py:
    def __init__(self, dim, lr=1.0, beta1=0.9, beta2=0.999, wd=1e-2):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8) + self.wd * self.lr * self.m


class AdaMOD_Py:
    def __init__(self, dim, lr=1.0, beta1=0.9, beta2=0.999, beta3=0.9995):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.t = 0
        self.s = 0.0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        gamma = self.lr / (np.sqrt(v_hat) + 1e-8)
        self.s = self.beta3 * self.s + (1 - self.beta3) * np.mean(gamma)
        lr = np.where(gamma < self.s, gamma, self.s)
        return lr * m_hat


class Muno_Py:
    def __init__(self, dim, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, ams=False):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ams = ams
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.vm = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        if self.ams:
            self.vm = np.maximum(self.vm, self.v)
            v_hat = self.vm / (1 - self.beta2**self.t)
        else:
            v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MunoW_Py:
    def __init__(
        self, dim, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-2, ams=False
    ):
        self.dim = dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.ams = ams
        self.m = np.zeros(dim, np.float32)
        self.v = np.zeros(dim, np.float32)
        self.vm = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        if self.ams:
            self.vm = np.maximum(self.vm, self.v)
            v_hat = self.vm / (1 - self.beta2**self.t)
        else:
            v_hat = self.v / (1 - self.beta2**self.t)
        return (
            self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * self.lr * self.m
        )


def zps_py(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    t = False
    if X.shape[-2] > X.shape[-1]:
        X = np.swapaxes(X, -2, -1)
        t = True
    X = X / (np.linalg.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.swapaxes(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if t:
        X = np.swapaxes(X, -2, -1)
    return X


class Muon_Py:
    def __init__(self, dim, lr=0.02, wd=0, mom=0.95, ns=5):
        self.dim = dim
        self.lr = lr
        self.wd = wd
        self.mom = mom
        self.ns = ns
        self.mb = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        if self.wd > 0:
            grad = grad + self.wd * self.mb
        self.mb = self.mom * self.mb + (1 - self.mom) * grad
        u = grad * (1 - self.mom) + self.mb * self.mom
        u = zps_py(u.reshape(1, -1), self.ns)
        u = u * max(1, 1 / self.dim) ** 0.5
        return -self.lr * u.flatten()


class AdamNS_Py:
    def __init__(self, dim, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, ns=5):
        self.dim = dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.ns = ns
        self.b1b = np.zeros(dim, np.float32)
        self.b2b = np.zeros(dim, np.float32)
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.b1b = self.b1 * self.b1b + (1 - self.b1) * grad
        self.b2b = self.b2 * self.b2b + (1 - self.b2) * grad**2
        u = (
            self.b1b
            / (1 - self.b1**self.t)
            / (np.sqrt(self.b2b / (1 - self.b2**self.t)) + self.eps)
        )
        if grad.ndim >= 2:
            u = zps_py(u.reshape(1, -1), self.ns)
        return self.lr * u.flatten()


from adam_cython import SGD, Adam, AdamW, AdaMOD, Muno, MunoW, Muon, AdamNS

np.random.seed(42)
dims = [10, 100, 1000, 10000]
n_iter = 1000

optimizer_configs = {
    "SGD": (SGD, SGD_Py, {"lr": 0.1}, {"lr": 0.1}),
    "Adam": (
        Adam,
        Adam_Py,
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999},
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999},
    ),
    "AdamW": (
        AdamW,
        AdamW_Py,
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01},
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "wd": 0.01},
    ),
    "AdaMOD": (
        AdaMOD,
        AdaMOD_Py,
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "beta3": 0.9995},
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "beta3": 0.9995},
    ),
    "Muno": (
        Muno,
        Muno_Py,
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999},
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999},
    ),
    "MunoW": (
        MunoW,
        MunoW_Py,
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01},
        {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "wd": 0.01},
    ),
    "Muon": (Muon, Muon_Py, {"lr": 0.02}, {"lr": 0.02}),
    "AdamNS": (AdamNS, AdamNS_Py, {"lr": 1e-3}, {"lr": 1e-3}),
}

print("=" * 100)
print("CYTHON OPTIMIZER PERFORMANCE COMPARISON")
print("=" * 100)

results = []
for dim in dims:
    print("\n=== Dimension: %6d ===" % dim)
    grad = np.random.randn(dim).astype(np.float32)
    header = "{:8s} {:>10s} {:>10s} {:>4s} {:>10s} {:>10s} {:>8s}".format(
        "Name", "MaxDiff", "RelDiff", "C", "Cython_ms", "Python_ms", "Speedup"
    )
    print(header)
    print("-" * 70)
    for name, (OC, OP, ckw, pkw) in optimizer_configs.items():
        oc = OC(dim, **ckw)
        op = OP(dim, **pkw)
        gc = oc.update(grad)
        gp = op.update(grad)
        md = np.abs(gc - gp).max()
        rd = md / (np.abs(gp).max() + 1e-10)
        c = "OK" if md < 1e-5 else "FAIL"
        for _ in range(5):
            oc.update(grad)
            op.update(grad)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            oc.update(grad)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n_iter):
            op.update(grad)
        tp = time.perf_counter() - t0
        line = "{:8s} {:10.2e} {:10.2e} {:>4s} {:10.1f} {:10.1f} {:8.2f}x".format(
            name, md, rd, c, tc * 1000, tp * 1000, tp / tc
        )
        print(line)
        results.append(
            {
                "optimizer": name,
                "dim": dim,
                "max_diff": md,
                "rel_diff": rd,
                "correct": c,
                "cython_ms": tc * 1000,
                "python_ms": tp * 1000,
                "speedup": tp / tc,
            }
        )

# Save results for report
import json

with open("benchmark_results.json", "w") as f:
    # Convert numpy types to Python native
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    json.dump(convert(results), f, indent=2)
