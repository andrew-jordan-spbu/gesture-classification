"""Признаки формы руки для классификации жестов.

Все признаки инвариантны к масштабу (нормировка на радиус ладони r),
повороту (углы считаются от собственной оси руки: ладонь -> запястье)
и зеркальному отражению (левая/правая рука, ладонь/тыл): используются
только беззнаковые углы, отсортированные наборы и симметричные комбинации.

Группы признаков:
  shape    — площадь, компактность, solidity, выпуклость, эксцентриситет
  hu       — моменты Ху (log), |h7| для инвариантности к отражению
  holes    — число и площадь отверстий (кольцо в «ОК»)
  defects  — дефекты выпуклости (впадины между пальцами)
  circles  — число пересечений маски с окружностями радиусов k*r
  fingers  — выступы-пальцы: углы от оси руки, длины, толщины, промежутки
  base     — основания пальцев: расстояния между ними и ширина (сомкнутые пальцы шире)
  axis     — протяжённость руки вдоль и поперёк оси запястья
  signature— статистические моменты сигнатуры границы r(phi)
  fourier  — Фурье-дескрипторы контура
  curv     — гистограмма кривизны контура
  skeleton — концевые точки и точки ветвления скелета (Чжан—Суэн)
"""
import cv2
import numpy as np

from segment import CUT_R, FOREARM_THICK, SLEEVE_DARK, background_brightness, far_end_dark, touches_border

R0 = 30.0              # радиус ладони после нормировки, px
CANVAS = 480           # размер нормированного холста (центр ладони в середине)
N_CONTOUR = 128        # точек на контуре после передискретизации
N_FD = 10              # число Фурье-дескрипторов (k = 2..11)
N_FING = 5             # максимум пальцев в признаках
CIRCLES = (1.4, 1.8, 2.2, 2.6, 3.0)
FINGER_MIN_REACH = 1.9  # палец должен доставать дальше этого (в r)
HOLE_MIN_AREA = 0.05    # минимальная площадь отверстия (в r^2)
WRIST_EXCLUDE_DEG = 45  # выступы ближе этого угла к запястью — не пальцы
WRIST_MIN_BASE = 0.8    # минимальная ширина основания обрубка запястья (в r)


# ---------------------------------------------------------------- геометрия
def wrist_direction(seg):
    """Единичный вектор от центра ладони к запястью (в координатах рабочего кадра)."""
    mask, removed = seg["mask"], seg["removed"]
    c = np.array(seg["center"], float)
    r = seg["radius"]

    def toward(pts):
        v = pts.mean(axis=0) - c
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    if removed.any():  # отрезанное предплечье
        ys, xs = np.nonzero(removed)
        pts = np.stack([xs, ys], 1).astype(float)
        near = pts[np.linalg.norm(pts - c, axis=1) < 3 * r]
        u = toward(near if len(near) else pts)
        if u is not None:
            return u
    if touches_border(mask):  # рука уходит за край кадра
        edge = np.zeros_like(mask)
        edge[:2], edge[-2:], edge[:, :2], edge[:, -2:] = mask[:2], mask[-2:], mask[:, :2], mask[:, -2:]
        ys, xs = np.nonzero(edge)
        u = toward(np.stack([xs, ys], 1).astype(float))
        if u is not None:
            return u
    # Рукав закрыл запястье, и рука не касается края кадра.
    rc = int(CUT_R * r)
    V, bg_v = background_brightness(seg["img"], seg["skin"])
    outside = mask.copy()
    cv2.circle(outside, (int(c[0]), int(c[1])), rc, 0, -1)
    ring = np.zeros_like(mask)
    cv2.circle(ring, (int(c[0]), int(c[1])), rc + 2, 255, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    wide, thin = [], []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.15 * r * r:
            continue
        part = lab == i
        ys, xs = np.nonzero(part)
        pts = np.stack([xs, ys], 1).astype(float)
        base = np.count_nonzero(part & (ring > 0)) / 3.0 / r
        if base >= WRIST_MIN_BASE:
            wide.append((base, far_end_dark(part, mask, V, bg_v, c, r), pts))
        else:
            thin.append(pts)
    # 1) широкий выступ, упирающийся в рукав, — обрубок запястья
    stubs = [w for w in wide if w[1]]
    if stubs:
        u = toward(max(stubs, key=lambda w: w[0])[2])
        if u is not None:
            return u
    # 2) рукав прямо у ладони: самый длинный участок, где маска граничит с тёмным
    k = max(3, int(0.3 * r)) | 1
    band = (cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0) & (mask == 0) & (V < SLEEVE_DARK * bg_v)
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    if nb > 1:
        j = 1 + int(np.argmax(bst[1:, cv2.CC_STAT_AREA]))
        if bst[j, cv2.CC_STAT_AREA] >= 0.8 * r * k:   # контакт длиной не меньше ~0.8r
            ys, xs = np.nonzero(blab == j)
            u = toward(np.stack([xs, ys], 1).astype(float))
            if u is not None:
                return u
    # 3) широкий выступ упирается в фон — сомкнутые пальцы, запястье напротив
    if wide:
        u = toward(max(wide, key=lambda w: w[0])[2])
        if u is not None:
            return -u
    # 4) только тонкие пальцы — запястье напротив их среднего направления
    dirs = [toward(p) for p in thin]
    dirs = [d for d in dirs if d is not None]
    if dirs:
        v = np.sum(dirs, axis=0)
        if np.linalg.norm(v) > 1e-6:
            return -v / np.linalg.norm(v)
    return np.array([0.0, 1.0])  # ничего не видно: считаем, что рука снизу


def normalize(seg):
    """Маска на холсте CANVAS x CANVAS: центр ладони в середине, r = R0."""
    s = R0 / seg["radius"]
    cx, cy = seg["center"]
    M = np.array([[s, 0, CANVAS / 2 - s * cx], [0, s, CANVAS / 2 - s * cy]], np.float32)
    m = cv2.warpAffine(seg["mask"], M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return m


def outer_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)


def resample(cnt, n=N_CONTOUR):
    """Равномерная по длине дуги передискретизация замкнутого контура."""
    closed = np.vstack([cnt, cnt[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, t[-1], n, endpoint=False)
    return np.stack([np.interp(tt, t, closed[:, 0]), np.interp(tt, t, closed[:, 1])], 1)


# ------------------------------------------------------------ скелетизация
def zhang_suen(mask):
    """Скелетизация Чжана—Суэна (два подцикла с условиями на соседей P2..P9)."""
    img = (mask > 0).astype(np.uint8)
    img = np.pad(img, 1)
    while True:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int32) for x in nb)                         # N(p1)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int32)  # переходы 0->1
                    for i in range(8))
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            kill = (P[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                img[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


def skeleton_points(skel):
    s = skel.astype(np.uint8)
    nb = cv2.filter2D(s, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - s
    ends = (s == 1) & (nb == 1)
    branches = (s == 1) & (nb >= 3)
    return ends, branches


# ---------------------------------------------------------------- признаки
def finger_info(m, u):
    """Выступы-пальцы на нормированной маске. Возвращает список словарей."""
    C = np.array([CANVAS / 2, CANVAS / 2])
    fwd = -u  # направление «от запястья к пальцам»
    outside = m.copy()
    cv2.circle(outside, (int(C[0]), int(C[1])), int(CUT_R * R0), 0, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    fingers = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.15 * R0 * R0:
            continue
        part = np.where(lab == i, 255, 0).astype(np.uint8)
        ys, xs = np.nonzero(part)
        pts = np.stack([xs, ys], 1).astype(float) - C
        d = np.linalg.norm(pts, axis=1)
        tip = pts[np.argmax(d)]
        reach = d.max() / R0
        thick = 2 * cv2.distanceTransform(part, cv2.DIST_L2, 5).max() / R0
        cosw = np.dot(tip / np.linalg.norm(tip), u)
        if reach < FINGER_MIN_REACH or cosw > np.cos(np.radians(WRIST_EXCLUDE_DEG)):
            continue
        ang = np.arctan2(fwd[0] * tip[1] - fwd[1] * tip[0], np.dot(fwd, tip))  # знаковый угол от fwd
        # основание пальца: где выступ пересекает окружность CUT_R*R0
        base_px = pts[d < CUT_R * R0 + 4]
        base = base_px.mean(axis=0) if len(base_px) else tip * (CUT_R * R0 / np.linalg.norm(tip))
        base_ang = np.arctan2(fwd[0] * base[1] - fwd[1] * base[0], np.dot(fwd, base))
        fingers.append({"angle": ang, "reach": reach, "thick": thick, "tip": tip + C,
                        "base": base + C, "base_angle": base_ang, "base_width": len(base_px) / 4.0 / R0,
                        "area": st[i, cv2.CC_STAT_AREA] / R0 ** 2})
    return fingers


def circle_crossings(m, k):
    C = CANVAS / 2
    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    xs = np.clip((C + k * R0 * np.cos(phi)).astype(int), 0, CANVAS - 1)
    ys = np.clip((C + k * R0 * np.sin(phi)).astype(int), 0, CANVAS - 1)
    v = m[ys, xs] > 0
    return int(np.sum(v & ~np.roll(v, 1)))  # число участков маски на окружности


def moments_stats(x):
    mu, sd = x.mean(), x.std()
    if sd < 1e-9:
        return [mu, 0.0, 0.0, 0.0]
    z = (x - mu) / sd
    return [mu, sd, float(np.mean(z ** 3)), float(np.mean(z ** 4))]


def extract(seg, return_debug=False):
    """Вектор признаков (np.ndarray) и их имена. seg — результат segment.segment()."""
    u = wrist_direction(seg)
    m = normalize(seg)
    C = np.array([CANVAS / 2, CANVAS / 2])
    f, names = [], []

    def add(name, val):
        names.append(name)
        f.append(float(val))

    # --- shape
    area = (m > 0).sum()
    cnt = outer_contour(m)
    per = cv2.arcLength(cnt.astype(np.float32).reshape(-1, 1, 2), True)
    hull = cv2.convexHull(cnt.astype(np.float32).reshape(-1, 1, 2))
    hull_area, hull_per = cv2.contourArea(hull), cv2.arcLength(hull, True)
    mo = cv2.moments(m, binaryImage=True)
    cov = np.array([[mo["mu20"], mo["mu11"]], [mo["mu11"], mo["mu02"]]]) / mo["m00"]
    l1, l2 = sorted(np.linalg.eigvalsh(cov), reverse=True)
    add("shape_area", area / R0 ** 2)
    add("shape_compactness", per ** 2 / (4 * np.pi * area))
    add("shape_solidity", area / max(hull_area, 1))
    add("shape_convexity", hull_per / max(per, 1))
    add("shape_eccentricity", np.sqrt(max(0.0, 1 - l2 / l1)))
    # смещение центра масс от центра ладони вдоль оси руки (к пальцам) и поперёк (без знака)
    cm = np.array([mo["m10"], mo["m01"]]) / mo["m00"] - C
    add("shape_cm_forward", -np.dot(cm, u) / R0)
    add("shape_cm_lateral", abs(u[0] * cm[1] - u[1] * cm[0]) / R0)

    # --- hu
    hu = cv2.HuMoments(mo).ravel()
    hu[6] = abs(hu[6])
    for i, h in enumerate(hu):
        add(f"hu_{i + 1}", -np.sign(h) * np.log10(abs(h) + 1e-30))

    # --- holes
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = [cv2.contourArea(c) / R0 ** 2 for c, h in zip(cnts, hier[0]) if h[3] >= 0]
    holes = [a for a in holes if a >= HOLE_MIN_AREA]
    add("holes_count", len(holes))
    add("holes_max_area", max(holes, default=0.0))

    # --- defects (впадины между пальцами)
    ci = cnt.astype(np.int32).reshape(-1, 1, 2)
    hidx = cv2.convexHull(ci, returnPoints=False)
    depths = np.array([0.0])
    try:
        d = cv2.convexityDefects(ci, np.sort(hidx, axis=0))
        if d is not None:
            depths = d.reshape(-1, 4)[:, 3] / 256.0 / R0  # форма (N,1,4) или (N,4) в разных версиях
    except cv2.error:
        pass
    add("defects_n_deep", int((depths > 0.6).sum()))
    add("defects_max_depth", depths.max())
    add("defects_sum_depth", depths[depths > 0.3].sum())

    # --- circles
    for k in CIRCLES:
        add(f"circle_{k:.1f}", circle_crossings(m, k))

    # --- fingers
    fing = finger_info(m, u)
    add("fingers_n", len(fing))
    add("fingers_total_thick", sum(x["thick"] for x in fing))
    srt = sorted(fing, key=lambda x: abs(x["angle"]))
    for i in range(N_FING):
        x = srt[i] if i < len(srt) else None
        add(f"finger{i + 1}_absangle", abs(x["angle"]) if x else 0.0)
        add(f"finger{i + 1}_reach", x["reach"] if x else 0.0)
        add(f"finger{i + 1}_thick", x["thick"] if x else 0.0)
    angs = sorted(x["angle"] for x in fing)
    gaps = sorted(np.diff(angs), reverse=True) if len(angs) > 1 else []
    add("fingers_span", (angs[-1] - angs[0]) if len(angs) > 1 else 0.0)
    for i in range(N_FING - 1):
        add(f"fingers_gap{i + 1}", gaps[i] if i < len(gaps) else 0.0)
    # основания пальцев: у соседних пальцев («ножницы») основания рядом, у «козы» — разнесены
    # на ширину ладони, даже если кончики видны под маленьким углом
    by_base = sorted(fing, key=lambda x: x["base_angle"])
    bgaps = sorted((np.linalg.norm(b["base"] - a["base"]) / R0 for a, b in zip(by_base, by_base[1:])),
                   reverse=True)
    for i in range(N_FING - 1):
        add(f"base_gap{i + 1}", bgaps[i] if i < len(bgaps) else 0.0)
    add("base_span", (np.linalg.norm(by_base[-1]["base"] - by_base[0]["base"]) / R0) if len(by_base) > 1 else 0.0)
    add("base_angle_span", (by_base[-1]["base_angle"] - by_base[0]["base_angle"]) if len(by_base) > 1 else 0.0)
    widths = sorted((x["base_width"] for x in fing), reverse=True)
    for i in range(3):
        add(f"base_width{i + 1}", widths[i] if i < len(widths) else 0.0)
    # самый «боковой» палец (кандидат в большой) относительно остальных
    if fing:
        side = max(fing, key=lambda x: abs(x["angle"]))
        add("finger_side_absangle", abs(side["angle"]))
        add("finger_side_reach", side["reach"])
        add("finger_side_thick", side["thick"])
    else:
        add("finger_side_absangle", 0.0); add("finger_side_reach", 0.0); add("finger_side_thick", 0.0)

    # --- axis: протяжённость вдоль/поперёк оси руки
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(float) - C
    along = -(P @ u) / R0
    lat = (u[0] * P[:, 1] - u[1] * P[:, 0]) / R0
    add("axis_forward_max", along.max())
    add("axis_back_max", -along.min())
    lat_a, lat_b = lat.max(), -lat.min()
    add("axis_lateral_big", max(lat_a, lat_b))
    add("axis_lateral_small", min(lat_a, lat_b))

    # --- signature r(phi) и её моменты
    rs = resample(cnt)
    sig = np.linalg.norm(rs - C, axis=1) / R0
    for name, v in zip(("mean", "std", "skew", "kurt"), moments_stats(sig)):
        add(f"signature_{name}", v)
    add("signature_max", sig.max())

    # --- fourier (|Z_k| + |Z_-k|) / (|Z_1| + |Z_-1|)
    z = rs[:, 0] + 1j * rs[:, 1]
    Z = np.fft.fft(z)
    base = abs(Z[1]) + abs(Z[-1])
    for k in range(2, 2 + N_FD):
        add(f"fourier_{k}", (abs(Z[k]) + abs(Z[-k])) / base)

    # --- curv: гистограмма углов поворота контура
    step = 3
    a = rs - np.roll(rs, step, axis=0)
    b = np.roll(rs, -step, axis=0) - rs
    turn = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
    hist, _ = np.histogram(turn, bins=8, range=(-np.pi, np.pi))
    for i, h in enumerate(hist / len(turn)):
        add(f"curv_{i}", h)

    # --- skeleton (на уменьшенной вдвое маске для скорости)
    small = cv2.resize(m, (CANVAS // 2, CANVAS // 2), interpolation=cv2.INTER_AREA)
    small = np.where(small > 127, 1, 0).astype(np.uint8)
    skel = zhang_suen(small)
    ends, branches = skeleton_points(skel)
    ey, ex = np.nonzero(ends)
    far = np.hypot(ex * 2 - C[0], ey * 2 - C[1]) / R0 > FINGER_MIN_REACH
    add("skeleton_ends", int(ends.sum()))
    add("skeleton_ends_far", int(far.sum()))
    add("skeleton_branches", int(branches.sum()))
    add("skeleton_length", skel.sum() * 2 / R0)

    vec = np.array(f, dtype=np.float64)
    if return_debug:
        return vec, names, {"norm": m, "u": u, "fingers": fing, "skel": skel, "holes": len(holes)}
    return vec, names


def draw_features(dbg):
    """Отладочная картинка признаков на нормированной маске."""
    m = dbg["norm"]
    vis = cv2.cvtColor(m // 3, cv2.COLOR_GRAY2BGR)
    C = (CANVAS // 2, CANVAS // 2)
    sk = cv2.resize(dbg["skel"].astype(np.uint8) * 255, (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST)
    vis[sk > 0] = (120, 120, 120)
    cv2.circle(vis, C, int(R0), (255, 255, 255), 1)
    cv2.circle(vis, C, int(CUT_R * R0), (0, 200, 255), 1)
    w = (int(C[0] + 2.5 * R0 * dbg["u"][0]), int(C[1] + 2.5 * R0 * dbg["u"][1]))
    cv2.arrowedLine(vis, C, w, (0, 0, 255), 2, tipLength=0.2)  # к запястью
    for i, fg in enumerate(sorted(dbg["fingers"], key=lambda x: abs(x["angle"]))):
        t = (int(fg["tip"][0]), int(fg["tip"][1]))
        cv2.line(vis, C, t, (0, 255, 0), 1)
        cv2.circle(vis, t, 6, (0, 255, 0), -1)
        cv2.putText(vis, f"{np.degrees(abs(fg['angle'])):.0f}", (t[0] + 6, t[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, f"fingers={len(dbg['fingers'])} holes={dbg['holes']}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis
