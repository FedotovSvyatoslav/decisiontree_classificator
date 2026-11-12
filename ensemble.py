#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ансамблевая классификация (Census Income / Adult). Базовый классификатор - дерево решений
из decision_tree.py (собственная реализация).

Техники:
  bagging  - бэггинг: каждое дерево строится по бутстреп-выборке, решение - голосование;
  forest   - случайный лес: бэггинг + случайное подмножество из --max-features атрибутов
             в каждом узле (по умолчанию round(sqrt(14)) = 4);
  boosting - AdaBoost (SAMME для двух классов) с взвешенной бутстреп-выборкой на каждом
             шаге; базовые деревья ограничены по глубине (--max-depth, по умолчанию 3).

Примеры:
  # одиночный запуск: 50 деревьев, 80% обучающая выборка
  python ensemble.py --train adult.data --test adult.test --technique forest --n-estimators 50 --train-frac 0.8

  # пп. 2-3: число участников 50, 60, ..., 100; три техники; линия одного дерева из прошлого задания
  python ensemble.py --train adult.data --test adult.test --technique all --experiment \
                     --train-frac 0.8 --baseline results/experiment.csv

В режиме --experiment для каждой техники обучается ансамбль из максимального числа участников,
а качество для меньших размеров считается по первым N участникам. Для бэггинга и леса участники
независимы и одинаково распределены, для бустинга первые N участников - это ровно модель после
N шагов, поэтому результаты совпадают с раздельным обучением ансамблей разного размера.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_tree import load_csv, Encoder, DecisionTree, metrics, split

TECHNIQUES = ["bagging", "forest", "boosting"]
TECH_RU = {"bagging": "Бэггинг", "forest": "Случайный лес", "boosting": "Бустинг (AdaBoost)"}
N_ATTR = 14  # 6 числовых + 8 категориальных


class Ensemble:
    def __init__(self, technique, n_estimators, criterion="gini", max_depth=None,
                 min_samples_split=2, max_features=None, seed=42):
        assert technique in TECHNIQUES
        self.tech = technique
        self.T = n_estimators
        self.crit = criterion
        self.min_split = min_samples_split
        self.seed = seed
        if max_depth is None and technique == "boosting":
            max_depth = 3
        self.max_depth = max_depth
        if technique == "forest" and max_features is None:
            max_features = int(round(np.sqrt(N_ATTR)))
        self.max_features = max_features if technique == "forest" else None

    def _new_tree(self, i):
        return DecisionTree(self.crit, self.max_depth, self.min_split,
                            self.max_features, np.random.RandomState(self.seed + 1000 + i))

    def fit(self, Xn, Xc, y):
        n = len(y)
        rng = np.random.RandomState(self.seed)
        self.trees, self.alphas = [], []
        if self.tech in ("bagging", "forest"):
            for i in range(self.T):
                idx = rng.randint(0, n, n)                       # бутстреп-выборка
                self.trees.append(self._new_tree(i).fit(Xn[idx], Xc[idx], y[idx]))
                self.alphas.append(1.0)
        else:
            w = np.full(n, 1.0 / n)
            attempts = 0
            while len(self.trees) < self.T and attempts < 5 * self.T:
                attempts += 1
                idx = rng.choice(n, size=n, p=w)                 # выборка с учётом весов
                tree = self._new_tree(len(self.trees)).fit(Xn[idx], Xc[idx], y[idx])
                wrong = tree.predict(Xn, Xc) != y
                err = float(w[wrong].sum())
                if err >= 0.5:                                    # слабый классификатор хуже случайного
                    w = np.full(n, 1.0 / n)
                    continue
                err = max(err, 1e-10)
                alpha = 0.5 * np.log((1 - err) / err)
                w = w * np.exp(np.where(wrong, alpha, -alpha))
                w /= w.sum()
                self.trees.append(tree)
                self.alphas.append(alpha)
        self.alphas = np.array(self.alphas)
        return self

    def member_predictions(self, Xn, Xc):
        return np.array([t.predict(Xn, Xc) for t in self.trees])

    def vote(self, P, k=None):
        """Решение ансамбля из первых k участников по матрице предсказаний P (участники x объекты)."""
        k = len(P) if k is None else k
        if self.tech == "boosting":
            s = (self.alphas[:k, None] * (2 * P[:k] - 1)).sum(axis=0)
            return (s > 0).astype(int)
        return (P[:k].sum(axis=0) * 2 > k).astype(int)

    def predict(self, Xn, Xc):
        return self.vote(self.member_predictions(Xn, Xc))


def plot_results(rows, baseline, path, title_extra=""):
    df = pd.DataFrame(rows)
    names = [("accuracy", "Accuracy"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F-мера")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (m, title) in zip(axs.ravel(), names):
        for tech, g in df.groupby("technique", sort=False):
            ax.plot(g["n_estimators"], g[m], marker="o", label=TECH_RU[tech])
        if baseline is not None:
            ax.axhline(baseline[m], color="k", ls="--", lw=1.3, label="Одно дерево (Gini)")
        ax.set_title(title)
        ax.set_xlabel("Количество участников ансамбля")
        ax.set_ylabel(title)
        ax.set_xticks(sorted(df["n_estimators"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Качество классификации в зависимости от размера ансамбля" + title_extra)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_baseline(path, frac, crit):
    if not path:
        return None
    b = pd.read_csv(path)
    b = b[(np.isclose(b["train_frac"], frac)) & (b["criterion"] == crit)]
    return None if b.empty else b.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser(description="Ансамблевая классификация: Census Income (Adult)")
    ap.add_argument("--train", required=True, help="adult.data")
    ap.add_argument("--test", default=None, help="adult.test")
    ap.add_argument("--technique", default="forest", choices=TECHNIQUES + ["all"])
    ap.add_argument("--n-estimators", type=int, default=50, help="число участников (одиночный запуск)")
    ap.add_argument("--experiment", action="store_true", help="серия: размер ансамбля из --sizes")
    ap.add_argument("--sizes", default="50,60,70,80,90,100")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="доля обучающей выборки; 1.0 = весь --train, тест = --test")
    ap.add_argument("--criterion", default="gini", choices=["infogain", "gainratio", "gini"])
    ap.add_argument("--max-depth", type=int, default=None,
                    help="глубина базовых деревьев (бустинг: по умолчанию 3, остальные: без ограничения)")
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=None, help="случайный лес: атрибутов в узле")
    ap.add_argument("--baseline", default=None,
                    help="experiment.csv из задания с одним деревом (для линии на графиках)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_ensemble")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df = load_csv(args.train)
    test_df = load_csv(args.test) if args.test else None
    full_df = pd.concat([train_df, test_df], ignore_index=True) if test_df is not None else train_df
    if args.train_frac >= 1.0 and test_df is not None:
        tr, te = train_df, test_df
    else:
        tr, te = split(full_df, args.train_frac, args.seed)   # то же разбиение, что в задании с деревом
    enc = Encoder().fit(tr)
    Xn, Xc, y = enc.transform(tr)
    Tn, Tc, ty = enc.transform(te)
    techs = TECHNIQUES if args.technique == "all" else [args.technique]
    print(f"обучающая выборка: {len(tr)}, тестовая: {len(te)}")

    sizes = [int(s) for s in args.sizes.split(",")] if args.experiment else [args.n_estimators]
    rows = []
    for tech in techs:
        ens = Ensemble(tech, max(sizes), args.criterion, args.max_depth, args.min_samples_split,
                       args.max_features, args.seed)
        t0 = time.time()
        ens.fit(Xn, Xc, y)
        fit_time = time.time() - t0
        P = ens.member_predictions(Tn, Tc)
        for k in sizes:
            res = metrics(ty, ens.vote(P, k))
            res.update({"technique": tech, "n_estimators": k, "n_train": len(tr), "n_test": len(te),
                        "train_frac": args.train_frac, "criterion": args.criterion,
                        "fit_time_s_total": round(fit_time, 1)})
            rows.append(res)
            print(f"{tech:9s} N={k:3d}  acc={res['accuracy']:.4f} prec={res['precision']:.4f} "
                  f"rec={res['recall']:.4f} f1={res['f1']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "ensemble_results.csv"), index=False)

    if args.experiment:
        base = load_baseline(args.baseline, args.train_frac, args.criterion)
        if args.baseline and base is None:
            print("предупреждение: в baseline нет строки с такой долей обучающей выборки и критерием")
        plot_results(rows, base, os.path.join(args.out, "metrics_vs_ensemble_size.png"),
                     f" (обучающая выборка {args.train_frac:.0%})")


if __name__ == "__main__":
    main()
