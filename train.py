#!/usr/bin/env python3
"""Признаки по датасету, оценка классификаторов и обучение итоговой модели.

    python train.py              # признаки (с кэшем), оценка, сохранение model.joblib
    python train.py --rebuild    # пересчитать признаки заново

Оценка двумя способами:
  1. Stratified 5-fold (3 повтора) — обычная кросс-валидация. Соседние кадры одной серии
     очень похожи, поэтому эта оценка оптимистична.
  2. «Новый ракурс»: обучаем на трёх комбинациях рука x сторона, тестируем на четвёртой,
     которую модель не видела (например, обучение без левой руки тыльной стороной).
     Это проверка инвариантности признаков к отражению.

Обучение идёт на подклассах (варианты «или» и разновидности «другого жеста» — отдельные
классы), а ответ объединяется обратно в 11 классов. Так «другой жест» не превращается
в одно большое размытое облако, которое затягивает соседние классы.
"""
import argparse
import time
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import features as F
import segment as S

CLASS_NAMES = {
    "kamen": "Камень", "bumaga": "Бумага", "nozhnitsy": "Ножницы", "klass": "Класс",
    "ok": "ОК", "koza": "Коза", "pozvoni": "Позвони мне", "pistolet": "Пистолет",
    "tri": "Три пальца", "fiksiki": "Фиксики", "drugoy": "Другой жест",
}
CLASS_ORDER = list(CLASS_NAMES)

# у редких подклассов («другой: средний палец») во внутреннем подборе параметров SVM
# бывает 2 кадра на фолд — sklearn об этом предупреждает, на результат это не влияет
warnings.filterwarnings("ignore", message="The least populated class in y")


def build_features(root: Path, cache: Path, rebuild: bool):
    if cache.exists() and not rebuild:
        return pd.read_csv(cache)
    rows, names = [], None
    files = sorted(root.glob("*/*.jpg"))
    t0 = time.time()
    for f in files:
        cls, var, hand, side = f.stem.split("__")[:4]
        seg = S.segment(cv2.imread(str(f)))
        if seg is None:
            print(f"  рука не найдена: {f.name}")
            continue
        vec, names = F.extract(seg)
        rows.append([f.name, cls, var, hand, side, *vec])
    df = pd.DataFrame(rows, columns=["file", "cls", "variant", "hand", "side", *names])
    df.to_csv(cache, index=False)
    print(f"Признаки: {len(df)} кадров x {len(names)} признаков, {time.time() - t0:.1f} с")
    return df


def models():
    svm = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
                       {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.003, 0.01, 0.03]},
                       cv=3, n_jobs=-1)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights="distance"))
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)
    return {"SVM (RBF)": svm, "kNN (k=3)": knn, "Случайный лес": rf}


def to_class(pred):
    """Подкласс «koza:TIL» -> класс «koza»."""
    return np.array([p.split(":")[0] for p in pred], dtype=object)


def cv_stratified(model, X, y, ysub, repeats=3):
    pred_all, accs, f1s = np.empty((repeats, len(y)), dtype=object), [], []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y):
            m = clone(model).fit(X[tr], ysub[tr])
            pred_all[rep, te] = to_class(m.predict(X[te]))
        accs.append(accuracy_score(y, pred_all[rep]))
        f1s.append(f1_score(y, pred_all[rep], average="macro"))
    return np.mean(accs), np.std(accs), np.mean(f1s), pred_all[0]


def cv_new_view(model, X, y, ysub, view):
    pred = np.empty(len(y), dtype=object)
    per_view = {}
    for v in np.unique(view):
        te = view == v
        m = clone(model).fit(X[~te], ysub[~te])
        pred[te] = to_class(m.predict(X[te]))
        per_view[v] = accuracy_score(y[te], pred[te])
    return accuracy_score(y, pred), per_view


def save_confusion(y, pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
    names = [CLASS_NAMES[c] for c in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("На самом деле")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = build_features(Path(args.data), out / "features.csv", args.rebuild)
    feat_cols = list(df.columns[5:])
    X = df[feat_cols].to_numpy(float)
    y = df["cls"].to_numpy()
    ysub = np.where(df["variant"] == "-", df["cls"], df["cls"] + ":" + df["variant"]).astype(object)
    view = (df["hand"] + "-" + df["side"]).to_numpy()
    print(f"\nКадров: {len(y)}, признаков: {X.shape[1]}")
    print(df["cls"].value_counts().reindex(CLASS_ORDER).to_string(), "\n")

    report = []
    best_name, best_acc, best_pred = None, -1, None
    for name, model in models().items():
        acc, sd, f1, pred = cv_stratified(model, X, y, ysub)
        acc_v, per_view = cv_new_view(model, X, y, ysub, view)
        pv = ", ".join(f"{k}: {v:.2f}" for k, v in per_view.items())
        line = (f"{name:15} 5-fold: точность {acc:.3f} ± {sd:.3f}, macro-F1 {f1:.3f} | "
                f"новый ракурс: {acc_v:.3f} ({pv})")
        print(line)
        report.append(line)
        if acc > best_acc:
            best_name, best_acc, best_pred = name, acc, pred

    save_confusion(y, best_pred, out / "confusion.png")
    print(f"\nЛучшая модель: {best_name}. Матрица ошибок: {out / 'confusion.png'}")
    errs = df.loc[best_pred != y, ["file"]].assign(pred=best_pred[best_pred != y])
    errs.to_csv(out / "errors.csv", index=False)
    print(f"Ошибок: {len(errs)} (список в {out / 'errors.csv'})")

    # вклад групп признаков: лучшая модель без каждой группы
    print(f"\nВклад групп признаков ({best_name}, 5-fold, без группы; со всеми: {best_acc:.3f}):")
    report.append(f"Вклад групп ({best_name}), со всеми признаками: {best_acc:.3f}")
    groups = sorted({c.split("_")[0].rstrip("0123456789") for c in feat_cols})
    for g in groups:
        keep = [i for i, c in enumerate(feat_cols) if c.split("_")[0].rstrip("0123456789") != g]
        acc, _, _, _ = cv_stratified(models()[best_name], X[:, keep], y, ysub)
        line = f"  без {g:10} -> {acc:.3f}"
        print(line)
        report.append(line)

    final = models()[best_name].fit(X, ysub)
    joblib.dump({"model": final, "features": feat_cols, "classes": CLASS_ORDER,
                 "names": CLASS_NAMES, "model_name": best_name}, "model.joblib")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nИтоговая модель сохранена: model.joblib")


if __name__ == "__main__":
    main()
