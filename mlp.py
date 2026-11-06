"""Нейросеть (MLP) на чистом numpy: обучение и прогноз. Без torch — ставить ничего не нужно.

Одна сеть на все 4 инструмента и все 3 горизонта сразу:
  вход  = признаки (обрезка по квантилям 0.1%/99.9% трейна → стандартизация) + one-hot инструмента
  выход = 3 числа (500 мс, 2 с, 8 с); таргет каждого дня делится на его std (как у LightGBM)
  лосс  = Huber (delta 1), пропуски таргета маскируются
  слои  = SiLU, по умолчанию 256 → 128 → 64; Adam; ранняя остановка по средней дневной
          корреляции на последних 10% дней трейна (валидация не трогается)

Прогноз — float64 и фиксированными кусками по CHUNK строк от начала дня (последний кусок
дополняется нулями). Поэтому значение в строке t не зависит от того, сколько строк после неё:
обрезка дня в момент T не меняет прогнозов до T (проверяет causality_check.py).
"""
import numpy as np

CHUNK = 8192


def silu(z):
    return z / (1.0 + np.exp(-z))


class MLPNumpy:
    """Готовая сеть для прогноза (то, что лежит в mlp.npz)."""

    def __init__(self, W, b, lo, hi, mean, std, n_inst):
        self.W = [np.asarray(w, np.float64) for w in W]
        self.b = [np.asarray(x, np.float64) for x in b]
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.mean, self.std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
        self.n_inst = int(n_inst)

    # ---------------------------------------------------------------- сохранение
    def save(self, path):
        d = {"lo": self.lo, "hi": self.hi, "mean": self.mean, "std": self.std,
             "n_inst": np.array(self.n_inst), "n_layers": np.array(len(self.W))}
        for k, (w, x) in enumerate(zip(self.W, self.b)):
            d[f"W{k}"], d[f"b{k}"] = w, x
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = int(z["n_layers"])
        return cls([z[f"W{k}"] for k in range(n)], [z[f"b{k}"] for k in range(n)],
                   z["lo"], z["hi"], z["mean"], z["std"], int(z["n_inst"]))

    # ---------------------------------------------------------------- прогноз
    def prep(self, X, inst_idx):
        X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Z = (np.clip(X, self.lo, self.hi) - self.mean) / self.std
        oh = np.zeros((len(Z), self.n_inst))
        oh[:, inst_idx] = 1.0
        return np.hstack([Z, oh])

    def _forward(self, H):
        for k, (w, x) in enumerate(zip(self.W, self.b)):
            H = H @ w + x
            if k < len(self.W) - 1:
                H = silu(H)
        return H

    def predict(self, X, inst_idx):
        """→ (n, 3). Куски по CHUNK строк от начала массива, последний дополняется нулями."""
        H = self.prep(X, inst_idx)
        n = len(H)
        out = np.empty((n, self.W[-1].shape[1]))
        for s in range(0, n, CHUNK):
            part = H[s:s + CHUNK]
            if len(part) < CHUNK:
                part = np.vstack([part, np.zeros((CHUNK - len(part), H.shape[1]))])
            out[s:s + CHUNK] = self._forward(part)[:min(CHUNK, n - s)]
        return out


# ==================================================================== обучение
def _huber_grad(r, delta):
    return np.clip(r, -delta, delta)


def _huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def fit_prep(X, rng, n_sample=300_000):
    """Квантили 0.1%/99.9%, среднее и std после обрезки — по случайной подвыборке трейна."""
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    S = np.nan_to_num(X[idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.quantile(S, 0.001, axis=0), np.quantile(S, 0.999, axis=0)
    C = np.clip(S, lo, hi)
    mean, std = C.mean(0), C.std(0)
    std[~(std > 1e-12)] = 1.0
    return lo, hi, mean, std


def loss_grads(W, B, x, y, m, delta=1.0, weight_decay=0.0):
    """Huber-лосс (сумма по точкам с таргетом) и градиенты средней потери по W, B."""
    hs, zs, h = [x], [], x
    for j in range(len(W)):
        z = h @ W[j] + B[j]
        zs.append(z)
        h = silu(z) if j < len(W) - 1 else z
        hs.append(h)
    r = (h - y) * m
    denom = max(float(m.sum()), 1.0)
    loss = float(_huber(r, delta).sum())
    g = _huber_grad(r, delta) * m / denom
    gWs, gBs = [None] * len(W), [None] * len(W)
    for j in range(len(W) - 1, -1, -1):
        gWs[j] = hs[j].T @ g + weight_decay * W[j]
        gBs[j] = g.sum(0)
        if j > 0:
            g = g @ W[j].T
            z = zs[j - 1]
            sg = 1.0 / (1.0 + np.exp(-z))
            g = g * (sg * (1.0 + z * (1.0 - sg)))
    return loss, gWs, gBs, denom


def train_mlp(Xtr, Ytr, Mtr, Itr, es, n_inst, hidden=(256, 128, 64), lr=1e-3, batch=4096,
              epochs=15, patience=3, weight_decay=1e-5, delta=1.0, seed=0, log=print):
    """Xtr (n,k) float32, Ytr (n,3) нормированный таргет (пропуски = 0), Mtr (n,3) маска 0/1,
    Itr (n,) номер инструмента. es — функция(net) → средняя дневная корреляция на днях
    ранней остановки (или None). → MLPNumpy."""
    rng = np.random.default_rng(seed)
    lo, hi, mean, std = fit_prep(Xtr, rng)
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    k = Xtr.shape[1]
    sizes = [k + n_inst, *hidden, Ytr.shape[1]]
    W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
    W[-1] *= 0.1
    B = [np.zeros(b, np.float32) for b in sizes[1:]]
    mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
    mB = [np.zeros_like(x) for x in B]; vB = [np.zeros_like(x) for x in B]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    eye = np.eye(n_inst, dtype=np.float32)

    def batch_input(idx):
        x = np.nan_to_num(Xtr[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x = (np.clip(x, lo32, hi32) - mean32) / std32
        return np.hstack([x, eye[Itr[idx]]]).astype(np.float32)

    def snapshot():
        return MLPNumpy(W, B, lo, hi, mean, std, n_inst)

    best, best_net, bad, cur_lr = -np.inf, snapshot(), 0, lr
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = np.sort(perm[s:s + batch])
            x, y, m = batch_input(idx), Ytr[idx], Mtr[idx]
            loss, gWs, gBs, denom = loss_grads(W, B, x, y, m, delta, weight_decay)
            tot += loss; cnt += denom
            t += 1
            for j in range(len(W)):
                gW, gB = gWs[j], gBs[j]
                for P, G, M_, V_ in ((W[j], gW, mW[j], vW[j]), (B[j], gB, mB[j], vB[j])):
                    M_ *= b1; M_ += (1 - b1) * G
                    V_ *= b2; V_ += (1 - b2) * G * G
                    P -= (cur_lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)) * M_ / (np.sqrt(V_) + eps)
        net = snapshot()
        score = es(net) if es is not None else float(ep)   # без дней остановки — берём последнюю эпоху
        log(f"     эпоха {ep + 1}: лосс {tot / max(cnt, 1):.4f}, корреляция на днях остановки {score:.4f}, lr {cur_lr:.1e}")
        if score > best + 1e-5:
            best, best_net, bad = score, net, 0
        else:
            bad += 1
            cur_lr *= 0.5
            if bad >= patience:
                break
    return best_net, best
