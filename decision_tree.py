# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship",
       "race", "sex", "native-country"]
POS = ">50K"
CLASS_NAMES = ["<=50K", ">50K"]
CRITERIA = ["infogain", "gainratio", "gini"]


# ----------------------------------------------------------------- данные
def load_csv(path):
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True,
                     comment="|", dtype=str)
    df = df.dropna(how="all")
    for c in COLS:
        df[c] = df[c].astype(str).str.strip()
    df["income"] = df["income"].str.rstrip(".")
    for c in NUM:
        df[c] = pd.to_numeric(df[c])
    return df.reset_index(drop=True)


class Encoder:
    def fit(self, df):
        self.cats = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CAT}
        self.inv = {c: {i: v for v, i in m.items()} for c, m in self.cats.items()}
        return self

    def transform(self, df):
        Xn = df[NUM].to_numpy(dtype=float)
        Xc = np.column_stack([df[c].map(self.cats[c]).fillna(-1).astype(int).to_numpy()
                              for c in CAT])
        y = (df["income"] == POS).astype(int).to_numpy()
        return Xn, Xc, y


# ----------------------------------------------------------------- критерии
def impurity(p, crit):
    p = np.asarray(p, dtype=float)
    if crit == "gini":
        return 2.0 * p * (1.0 - p)
    q = 1.0 - p
    lp = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    lq = np.where(q > 0, q * np.log2(np.where(q > 0, q, 1.0)), 0.0)
    return -(lp + lq)


class Node:
    __slots__ = ("n", "pos", "pred", "kind", "attr", "thr", "left", "right", "children")

    def __init__(self, n, pos):
        self.n, self.pos = n, pos
        self.pred = int(pos * 2 > n)
        self.kind = None          # None (лист) | 'num' | 'cat'
        self.attr = self.thr = self.left = self.right = self.children = None

    @property
    def is_leaf(self):
        return self.kind is None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        assert criterion in CRITERIA
        self.crit = criterion
        self.max_depth = max_depth
        self.min_split = min_samples_split

    def fit(self, Xn, Xc, y):
        self.Xn, self.Xc, self.y = Xn, Xc, y
        self.root = self._build(np.arange(len(y)), 0)
        del self.Xn, self.Xc, self.y
        return self

    # --- поиск разбиений
    def _best_numeric(self, v, ys, parent_imp):
        n = len(v)
        order = np.argsort(v, kind="stable")
        vs, yy = v[order], ys[order]
        c1 = np.cumsum(yy)
        valid = np.nonzero(vs[:-1] != vs[1:])[0]
        if len(valid) == 0:
            return None
        nl = valid + 1
        pl = c1[valid]
        nr = n - nl
        pr = c1[-1] - pl
        w = (nl / n) * impurity(pl / nl, self.crit) + (nr / n) * impurity(pr / nr, self.crit)
        k = int(np.argmin(w))
        gain = parent_imp - w[k]
        fl = nl[k] / n
        si = -(fl * np.log2(fl) + (1 - fl) * np.log2(1 - fl))
        thr = (vs[valid[k]] + vs[valid[k] + 1]) / 2.0
        return gain, gain / si, thr

    def _best_categorical(self, codes, ys, parent_imp):
        n = len(codes)
        K = int(codes.max()) + 1
        cnt = np.bincount(codes, minlength=K)
        pos = np.bincount(codes, weights=ys, minlength=K)
        nz = cnt > 0
        if nz.sum() < 2:
            return None
        frac = cnt[nz] / n
        w = np.sum(frac * impurity(pos[nz] / cnt[nz], self.crit))
        gain = parent_imp - w
        si = -np.sum(frac * np.log2(frac))
        return gain, gain / si, None

    def _build(self, idx, depth):
        ys = self.y[idx]
        n, pos = len(idx), int(ys.sum())
        node = Node(n, pos)
        if pos == 0 or pos == n or n < self.min_split or \
                (self.max_depth is not None and depth >= self.max_depth):
            return node
        parent_imp = float(impurity(pos / n, self.crit))
        cands = []  # (gain, ratio, kind, attr, thr)
        for j in range(self.Xn.shape[1]):
            r = self._best_numeric(self.Xn[idx, j], ys, parent_imp)
            if r and r[0] > 1e-9:
                cands.append((r[0], r[1], "num", j, r[2]))
        for j in range(self.Xc.shape[1]):
            r = self._best_categorical(self.Xc[idx, j], ys, parent_imp)
            if r and r[0] > 1e-9:
                cands.append((r[0], r[1], "cat", j, None))
        if not cands:
            return node
        if self.crit == "gainratio":
            avg = np.mean([c[0] for c in cands])
            pool = [c for c in cands if c[0] >= avg - 1e-12]
            best = max(pool, key=lambda c: c[1])
        else:
            best = max(cands, key=lambda c: c[0])
        _, _, kind, j, thr = best
        node.kind, node.attr = kind, j
        if kind == "num":
            node.thr = thr
            mask = self.Xn[idx, j] <= thr
            node.left = self._build(idx[mask], depth + 1)
            node.right = self._build(idx[~mask], depth + 1)
        else:
            col = self.Xc[idx, j]
            node.children = {int(v): self._build(idx[col == v], depth + 1)
                             for v in np.unique(col)}
        return node

    # --- предсказание
    def predict(self, Xn, Xc):
        out = np.empty(len(Xn), dtype=int)
        for i in range(len(Xn)):
            node = self.root
            while not node.is_leaf:
                if node.kind == "num":
                    node = node.left if Xn[i, node.attr] <= node.thr else node.right
                else:
                    nxt = node.children.get(int(Xc[i, node.attr]))
                    if nxt is None:
                        break
                    node = nxt
            out[i] = node.pred
        return out

    def stats(self):
        nodes = leaves = 0
        maxd = 0
        stack = [(self.root, 0)]
        while stack:
            nd, d = stack.pop()
            nodes += 1
            maxd = max(maxd, d)
            if nd.is_leaf:
                leaves += 1
            elif nd.kind == "num":
                stack += [(nd.left, d + 1), (nd.right, d + 1)]
            else:
                stack += [(c, d + 1) for c in nd.children.values()]
        return {"nodes": nodes, "leaves": leaves, "depth": maxd}


# ----------------------------------------------------------------- метрики
def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ----------------------------------------------------------------- визуализация
def draw_tree(tree, enc, path, title, viz_depth=3):
    pos_xy, edges, labels, colors = {}, [], {}, {}
    counter = [0]

    def children(nd):
        if nd.kind == "num":
            return [(nd.left, f"<= {nd.thr:.4g}"), (nd.right, f"> {nd.thr:.4g}")]
        return [(c, enc.inv[CAT[nd.attr]].get(v, str(v)))
                for v, c in sorted(nd.children.items())]

    def place(nd, d):
        key = id(nd)
        colors[key] = "#f6d3c4" if nd.pred == 1 else "#cfe3f6"
        share = nd.pos / nd.n
        if nd.is_leaf:
            labels[key] = f"{CLASS_NAMES[nd.pred]}\nn={nd.n}"
        elif d >= viz_depth:
            labels[key] = f"...\n{CLASS_NAMES[nd.pred]}\nn={nd.n}"
        else:
            name = NUM[nd.attr] if nd.kind == "num" else CAT[nd.attr]
            labels[key] = f"{name}\nn={nd.n}\n>50K: {share:.0%}"
        if nd.is_leaf or d >= viz_depth:
            pos_xy[key] = (counter[0], -d)
            counter[0] += 1
            return pos_xy[key][0]
        xs = []
        for ch, lab in children(nd):
            x = place(ch, d + 1)
            xs.append(x)
            edges.append((key, id(ch), lab))
        pos_xy[key] = (float(np.mean(xs)), -d)
        return pos_xy[key][0]

    place(tree.root, 0)
    width = max(10, 1.5 * counter[0])
    height = max(4, 2.3 * (viz_depth + 1))
    fig, ax = plt.subplots(figsize=(min(width, 60), height))
    for a, b, lab in edges:
        (x1, y1), (x2, y2) = pos_xy[a], pos_xy[b]
        ax.plot([x1, x2], [y1, y2], color="gray", lw=0.8, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, lab, fontsize=6, ha="center",
                color="#333", zorder=2)
    for key, (x, y) in pos_xy.items():
        ax.text(x, y, labels[key], ha="center", va="center", fontsize=7, zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", fc=colors[key], ec="gray", lw=0.6))
    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_experiment(rows, path):
    df = pd.DataFrame(rows)
    names = [("accuracy", "Accuracy"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F-мера")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (m, title) in zip(axs.ravel(), names):
        for crit, g in df.groupby("criterion"):
            ax.plot(g["train_frac"] * 100, g[m], marker="o", label=crit)
        ax.set_title(title)
        ax.set_xlabel("Доля обучающей выборки, %")
        ax.set_ylabel(title)
        ax.set_xticks(sorted(df["train_frac"].unique() * 100))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- эксперименты
def split(df, frac, seed):
    perm = np.random.RandomState(seed).permutation(len(df))
    cut = int(round(frac * len(df)))
    return df.iloc[perm[:cut]].reset_index(drop=True), df.iloc[perm[cut:]].reset_index(drop=True)


def run_once(crit, train_df, test_df, args):
    enc = Encoder().fit(train_df)
    Xn, Xc, y = enc.transform(train_df)
    t0 = time.time()
    tree = DecisionTree(crit, args.max_depth, args.min_samples_split).fit(Xn, Xc, y)
    fit_time = time.time() - t0
    res = {"criterion": crit, "n_train": len(train_df), "n_test": len(test_df),
           "fit_time_s": round(fit_time, 2), **tree.stats()}
    if len(test_df):
        Tn, Tc, ty = enc.transform(test_df)
        res.update(metrics(ty, tree.predict(Tn, Tc)))
    return tree, enc, res


def main():
    ap = argparse.ArgumentParser(description="Дерево решений: Census Income (Adult)")
    ap.add_argument("--train", required=True, help="файл набора данных (adult.data)")
    ap.add_argument("--test", default=None, help="тестовый файл (adult.test)")
    ap.add_argument("--criterion", default="gini", choices=CRITERIA + ["all"])
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="доля обучающей выборки; 1.0 = весь --train, тест = --test; "
                         "<1.0 = --train и --test объединяются и делятся случайно")
    ap.add_argument("--experiment", action="store_true",
                    help="серия: доля обучающей выборки 0.6, 0.7, 0.8, 0.9")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--viz-depth", type=int, default=3,
                    help="глубина отображаемой части дерева на рисунке")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df = load_csv(args.train)
    test_df = load_csv(args.test) if args.test else None
    full_df = pd.concat([train_df, test_df], ignore_index=True) if test_df is not None else train_df
    crits = CRITERIA if args.criterion == "all" else [args.criterion]

    if args.experiment:
        rows = []
        for frac in (0.6, 0.7, 0.8, 0.9):
            tr, te = split(full_df, frac, args.seed)
            for crit in crits:
                tree, enc, res = run_once(crit, tr, te, args)
                res["train_frac"] = frac
                rows.append(res)
                print(f"{crit:9s} {frac:.0%}:{1 - frac:.0%}  acc={res['accuracy']:.4f} "
                      f"prec={res['precision']:.4f} rec={res['recall']:.4f} "
                      f"f1={res['f1']:.4f} nodes={res['nodes']}")
                draw_tree(tree, enc, os.path.join(args.out, f"tree_{crit}_{int(frac*100)}.png"),
                          f"Дерево решений ({crit}), обучающая выборка {frac:.0%}", args.viz_depth)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "experiment.csv"), index=False)
        plot_experiment(rows, os.path.join(args.out, "metrics_vs_split.png"))
        return

    results = []
    for crit in crits:
        if args.train_frac >= 1.0:
            tr, te = train_df, (test_df if test_df is not None else train_df.iloc[0:0])
        else:
            tr, te = split(full_df, args.train_frac, args.seed)
        tree, enc, res = run_once(crit, tr, te, args)
        res["train_frac"] = args.train_frac
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))
        draw_tree(tree, enc, os.path.join(args.out, f"tree_{crit}.png"),
                  f"Дерево решений ({crit}), обучение на {args.train_frac:.0%} данных",
                  args.viz_depth)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
