"""GRU на последовательностях признаков: обучение (torch, GPU если есть) и прогноз (чистый numpy).

Последовательность — те же признаки в моменты t, t-2.5 с, t-5 с, … (шаг STRIDE=5 строк 500-мс сетки:
в кэше трейна как раз каждая 5-я точка). Одна сеть на 4 инструмента (one-hot) и 3 горизонта.
  вход  = признаки (обрезка по квантилям 0.1%/99.9% трейна → стандартизация, как у mlp) + one-hot
  слой  = GRU (hidden) → линейный выход на 3 горизонта; таргет дня делится на его std
  обучение: случайные куски длины burn+L из дней трейна, лосс Huber только на последних L шагах
            (первые burn шагов «разогревают» состояние); Adam, ранняя остановка по дневной
            корреляции на последних 10% дней трейна — считается ТЕМ ЖЕ numpy-прогнозом, что и в сдаче.

Прогноз дня (predict.py): строки сетки делятся на STRIDE переплетённых потоков (t mod 5), состояние
каждого потока идёт с начала дня. Входные проекции — float64 фиксированными кусками CHUNK строк от
начала дня, шаг рекуррентности — матрица фиксированной формы (STRIDE потоков). Значение в строке t
зависит только от строк ≤ t и не зависит от того, сколько строк после неё: обрезка в момент T не
меняет прогнозов до T (causality_check.py). torch для прогноза не нужен.
"""
import numpy as np

STRIDE = 5          # строк 500-мс сетки между соседними шагами последовательности (2.5 с)
CHUNK = 8192
FULL_DAY = 172_800


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class GRUNumpy:
    """Готовая сеть (то, что лежит в gru.npz). Веса в формате torch.nn.GRU: гейты r, z, n."""

    def __init__(self, W_ih, W_hh, b_ih, b_hh, W_out, b_out, lo, hi, mean, std, n_inst):
        f = lambda x: np.asarray(x, np.float64)
        self.W_ih, self.W_hh, self.b_ih, self.b_hh = f(W_ih), f(W_hh), f(b_ih), f(b_hh)
        self.W_out, self.b_out = f(W_out), f(b_out)
        self.lo, self.hi, self.mean, self.std = f(lo), f(hi), f(mean), f(std)
        self.n_inst = int(n_inst)
        self.H = self.W_hh.shape[1]

    def save(self, path):
        np.savez(path, W_ih=self.W_ih, W_hh=self.W_hh, b_ih=self.b_ih, b_hh=self.b_hh, W_out=self.W_out,
                 b_out=self.b_out, lo=self.lo, hi=self.hi, mean=self.mean, std=self.std,
                 n_inst=np.array(self.n_inst), stride=np.array(STRIDE))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        if int(z["stride"]) != STRIDE:
            raise SystemExit(f"{path}: сеть училась с шагом {int(z['stride'])}, а код считает с {STRIDE}")
        return cls(z["W_ih"], z["W_hh"], z["b_ih"], z["b_hh"], z["W_out"], z["b_out"],
                   z["lo"], z["hi"], z["mean"], z["std"], int(z["n_inst"]))

    def prep(self, X, inst_idx):
        X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Z = (np.clip(X, self.lo, self.hi) - self.mean) / self.std
        oh = np.zeros((len(Z), self.n_inst))
        oh[:, inst_idx] = 1.0
        return np.hstack([Z, oh])

    def predict(self, X, inst_idx, row_gap=None):
        """X — строки ОДНОГО дня подряд от начала дня. row_gap — расстояние между соседними строками X
        в 500-мс сетке: 1 (полный день: predict.py, валидация) или 5 (прореженные дни кэша трейна).
        По умолчанию: 1, если строк больше половины полного дня, иначе 5. → (n, 3)."""
        if row_gap is None:
            row_gap = 1 if len(X) > FULL_DAY // 2 else STRIDE
        if STRIDE % row_gap:
            raise ValueError(f"row_gap {row_gap} не делит шаг {STRIDE}")
        S = STRIDE // row_gap                         # число переплетённых потоков
        Z = self.prep(X, inst_idx)
        n = len(Z)
        G = np.empty((n, self.W_ih.shape[0]))         # входные проекции, кусками от начала дня
        for s in range(0, n, CHUNK):
            part = Z[s:s + CHUNK]
            if len(part) < CHUNK:
                part = np.vstack([part, np.zeros((CHUNK - len(part), Z.shape[1]))])
            G[s:s + CHUNK] = (part @ self.W_ih.T + self.b_ih)[:min(CHUNK, n - s)]
        H, Wt, bh = self.H, self.W_hh.T, self.b_hh
        steps = -(-n // S)
        pad = steps * S - n
        if pad:
            G = np.vstack([G, np.zeros((pad, G.shape[1]))])
        G = G.reshape(steps, S, -1)                   # шаг k: строки k*S … k*S+S-1 (потоки 0..S-1)
        h = np.zeros((S, H))
        Hs = np.empty((steps, S, H))
        for k in range(steps):
            gi = G[k]
            gh = h @ Wt + bh
            r = _sig(gi[:, :H] + gh[:, :H])
            z = _sig(gi[:, H:2 * H] + gh[:, H:2 * H])
            nn = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * nn + z * h
            Hs[k] = h
        out = Hs.reshape(steps * S, H)[:n] @ self.W_out.T + self.b_out
        return out


# ==================================================================== обучение (torch)
def train_gru(days, es, n_inst, hidden=64, burn=64, seq=64, batch=256, steps=1500, epochs=20, lr=2e-3,
              patience=3, delta=1.0, seed=0, log=print):
    """days — список (k, X float32 (T,F) сырые признаки дня, Yn (T,3) таргет / std дня с пропусками = 0,
    M (T,3) маска 0/1); строки — прореженная сетка (шаг STRIDE). es — функция(net) → средняя дневная
    корреляция на днях ранней остановки (или None). → (GRUNumpy лучшей эпохи, её оценка)."""
    import torch
    from mlp import fit_prep

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = burn + seq
    days = [d for d in days if len(d[1]) >= L]
    if not days:
        raise SystemExit("gru: нет дней длиннее burn+seq")
    samp = np.vstack([d[1][rng.choice(len(d[1]), min(len(d[1]), 3000), replace=False)] for d in days])
    lo, hi, mean, std = fit_prep(samp, rng)
    del samp
    F = days[0][1].shape[1]
    lo32, hi32, m32, s32 = (v.astype(np.float32) for v in (lo, hi, mean, std))
    eye = np.eye(n_inst, dtype=np.float32)
    Xn, Yn, Mn, Kn = [], [], [], []
    for k, X, Y, M in days:                           # нормированные дни храним в float16 — экономим память
        Z = (np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), lo32, hi32) - m32) / s32
        Xn.append(Z.astype(np.float16)); Yn.append(Y.astype(np.float32)); Mn.append(M.astype(np.float32))
        Kn.append(k)
    lens = np.array([len(x) - L + 1 for x in Xn], np.float64)
    pday = lens / lens.sum()
    log(f"     gru: дней {len(Xn)}, шагов {int(sum(len(x) for x in Xn)):,}, вход {F} + {n_inst}, "
        f"hidden {hidden}, окно {burn}+{seq}, устройство {dev}")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(F + n_inst, hidden, batch_first=True)
            self.out = torch.nn.Linear(hidden, 3)
            with torch.no_grad():
                self.out.weight.mul_(0.1)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.out(h)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def snapshot():
        g, o = net.gru, net.out
        c = lambda t: t.detach().cpu().double().numpy()
        return GRUNumpy(c(g.weight_ih_l0), c(g.weight_hh_l0), c(g.bias_ih_l0), c(g.bias_hh_l0),
                        c(o.weight), c(o.bias), lo, hi, mean, std, n_inst)

    def make_batch():
        di = rng.choice(len(Xn), batch, p=pday)
        xs, ys, ms = [], [], []
        for d in di:
            s = int(rng.integers(0, len(Xn[d]) - L + 1))
            oh = np.broadcast_to(eye[Kn[d]], (L, n_inst))
            xs.append(np.hstack([Xn[d][s:s + L].astype(np.float32), oh]))
            ys.append(Yn[d][s + burn:s + L]); ms.append(Mn[d][s + burn:s + L])
        return (torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys)).to(dev),
                torch.from_numpy(np.stack(ms)).to(dev))

    best, best_net, bad, cur_lr = -np.inf, snapshot(), 0, lr
    for ep in range(epochs):
        net.train()
        tot, cnt = 0.0, 0.0
        for _ in range(steps):
            x, y, m = make_batch()
            p = net(x)[:, burn:]
            r = (p - y) * m
            a = r.abs()
            loss_sum = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)).sum()
            denom = m.sum().clamp_min(1.0)
            opt.zero_grad()
            (loss_sum / denom).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss_sum.item(); cnt += denom.item()
        snap = snapshot()
        score = es(snap) if es is not None else float(ep)
        log(f"     эпоха {ep + 1}: лосс {tot / max(cnt, 1):.4f}, корреляция на днях остановки {score:.4f}, lr {cur_lr:.1e}")
        if score > best + 1e-5:
            best, best_net, bad = score, snap, 0
        else:
            bad += 1
            cur_lr *= 0.5
            for gopt in opt.param_groups:
                gopt["lr"] = cur_lr
            if bad >= patience:
                break
    return best_net, best
