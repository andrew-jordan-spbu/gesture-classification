"""Сегментация руки на кадре.

Шаги:
1. Маска кожи: порог Оцу по каналу Cr пространства YCrCb.
2. Морфология: раскрытие (шум) и закрытие (мелкие разрывы).
3. Выбор руки: самая крупная связная область кожи (рука обычно ближе к камере, чем лицо).
   Если крупных областей несколько, каскады Хаара (Виола—Джонс) ищут лицо, и область,
   пересекающаяся с лицом, исключается. Каскады есть в пакете opencv-contrib-python;
   без него остаётся правило «самая крупная область».
   Если рядом с рукой (щель до BRIDGE_GAP px) есть ещё крупная область кожи — кисть,
   отрезанная ремешком часов или браслетом, — области соединяются мостом.
4. Центр ладони и её радиус r: максимум distance transform. Если видно, откуда рука
   входит в кадр (край кадра или граница с тёмным рукавом), центр ищется по профилю
   толщины вдоль руки: от дальнего конца (кончиков пальцев) толщина растёт до ладони,
   падает на запястье и снова растёт на предплечье, которое ближе к камере может быть
   толще кисти. Ладонь — первый пик перед провалом запястья.
5. Отрезание предплечья. Вырезаем круг радиуса CUT_R*r вокруг центра ладони и смотрим
   на оставшиеся выступы. Пальцы тонкие, предплечье толстое (> FOREARM_THICK*r) и либо
   уходит за край кадра, либо упирается в рукав (за его концом темнее фона). Такой выступ
   удаляем.
   Если центр ладони попал на предплечье (ядро касается края, а вдали от края есть
   толстый выступ — это кисть), переносим центр в этот выступ и повторяем.
"""
import cv2
import numpy as np

WORK_W = 640            # ширина, до которой уменьшаем кадр
CR_MIN_THR = 135        # нижняя граница порога по Cr (защита, если руки нет)
MIN_AREA_FRAC = 0.005   # минимальная доля кадра для области-кандидата
CUT_R = 1.6             # радиус круга вокруг ладони, за которым ищем предплечье (в r)
FOREARM_THICK = 0.55    # выступ толще этого (в r) считаем предплечьем
SLEEVE_DARK = 0.75      # «рукав» — темнее, чем SLEEVE_DARK * медиана яркости фона
FACE_SECOND_FRAC = 0.3  # ищем лицо, если вторая область кожи >= этой доли от первой
FACE_OVERLAP = 0.3      # область — лицо, если такая её доля внутри найденного лица
FACE_MIN_NEIGHBORS = 8  # строже стандартных 5: профильный каскад иначе срабатывает на руках
BRIDGE_GAP = 20         # px: максимальная щель (ремешок часов), через которую соединяем
BRIDGE_MIN_FRAC = 0.4   # соединяем области не меньше этой доли от руки (часть за ремешком ~0.6, обрывки шарфа <= 0.21)
PROFILE_BIN = 4         # px: шаг профиля толщины вдоль руки
PROFILE_DIP = 0.88      # провал запястья: толщина ниже этой доли от пика ладони
CONTACT_MIN = 25        # px: минимальная длина границы с рукавом
SLEEVE_MIN_AREA = 0.02  # рукав — тёмная область не меньше этой доли кадра (не ремешок часов)
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_CASCADES = None


def face_cascades():
    """Каскады Хаара для лица анфас и в профиль (None, если OpenCV без contrib)."""
    global _CASCADES
    if _CASCADES is None:
        _CASCADES = []
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
                c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                if not c.empty():
                    _CASCADES.append((name, c))
    return _CASCADES


def detect_faces(bgr):
    """Прямоугольники лиц (x, y, w, h): анфас, профиль и профиль на отражённом кадре."""
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    w_img = gray.shape[1]
    boxes = []
    for name, casc in face_cascades():
        for (x, y, w, h) in casc.detectMultiScale(gray, 1.1, FACE_MIN_NEIGHBORS, minSize=(40, 40)):
            boxes.append((x, y, w, h))
        if "profile" in name:
            for (x, y, w, h) in casc.detectMultiScale(cv2.flip(gray, 1), 1.1, FACE_MIN_NEIGHBORS, minSize=(40, 40)):
                boxes.append((w_img - x - w, y, w, h))
    return boxes


def background_brightness(bgr, skin):
    """Канал яркости V и медианная яркость всего, что не кожа."""
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    rest = V[skin == 0]
    return V, float(np.median(rest)) if rest.size else 255.0


def far_end_dark(part, hand, V, bg_v, c, r):
    """Упирается ли выступ part в тёмное (рукав): яркость сразу за его дальним концом."""
    ys, xs = np.nonzero(part)
    reach = np.hypot(xs - c[0], ys - c[1]).max() / r
    band = (cv2.dilate(part.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (hand == 0)
    by, bx = np.nonzero(band)
    far = np.hypot(bx - c[0], by - c[1]) / r > reach - 0.5
    if not far.any():
        return False
    return float(V[by[far], bx[far]].mean()) < SLEEVE_DARK * bg_v


def resize_work(bgr, width=WORK_W):
    h, w = bgr.shape[:2]
    if w == width:
        return bgr
    return cv2.resize(bgr, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def skin_mask(bgr):
    """Бинарная маска кожи и использованный порог по Cr."""
    cr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 1]
    cr = cv2.GaussianBlur(cr, (5, 5), 0)
    thr, _ = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr, CR_MIN_THR)
    mask = np.where(cr > thr, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)
    return mask, thr


def select_hand(skin, bgr):
    """Область кожи, которая является рукой: самая крупная, но не лицо."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if n <= 1:
        return None
    order = [i for i in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
             if st[i, cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * skin.size]
    if not order:
        return None
    if len(order) > 1 and st[order[1], cv2.CC_STAT_AREA] >= FACE_SECOND_FRAC * st[order[0], cv2.CC_STAT_AREA]:
        faces = detect_faces(bgr)
        if faces:
            face = np.zeros(skin.shape, np.uint8)
            for (x, y, w, h) in faces:
                cv2.rectangle(face, (int(x - 0.2 * w), int(y - 0.2 * h)),
                              (int(x + 1.2 * w), int(y + 1.5 * h)), 255, -1)
            not_face = [i for i in order
                        if np.count_nonzero((lab == i) & (face > 0)) < FACE_OVERLAP * st[i, cv2.CC_STAT_AREA]]
            if not_face:
                order = not_face
    hand = np.where(lab == order[0], 255, 0).astype(np.uint8)
    return bridge_parts(hand, [np.where(lab == i, 255, 0).astype(np.uint8) for i in order[1:]
                               if st[i, cv2.CC_STAT_AREA] >= BRIDGE_MIN_FRAC * st[order[0], cv2.CC_STAT_AREA]])


def bridge_parts(hand, others):
    """Присоединяет к руке области, отделённые узкой щелью (ремешок часов, браслет)."""
    for part in others:
        d_hand = cv2.distanceTransform(np.where(hand > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        gap = float(d_hand[part > 0].min())
        if gap > BRIDGE_GAP:
            continue
        d_part = cv2.distanceTransform(np.where(part > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        bridge = (d_hand <= gap + 2) & (d_part <= gap + 2) & (d_hand + d_part <= gap + 3)
        hand = np.where((hand > 0) | (part > 0) | bridge, 255, 0).astype(np.uint8)
    return hand


def largest_component(mask):
    """Самая крупная связная область (или None, если ничего достаточно крупного)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < MIN_AREA_FRAC * mask.size:
        return None
    return np.where(lab == i, 255, 0).astype(np.uint8)


def palm(mask, search=None):
    """Центр ладони (x, y) и радиус r — максимум distance transform.

    search — необязательная маска области, в которой ищем максимум.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    score = dist if search is None else np.where(search > 0, dist, 0)
    _, _, _, c = cv2.minMaxLoc(score)
    return c, float(dist[c[1], c[0]])


def entry_candidates(hand, V, bg_v):
    """Где рука может входить в кадр.

    Если рука касается края кадра — это край. Иначе кандидаты — два геодезических конца
    руки (самая дальняя по маске точка от ладони и самая дальняя от неё): кончики пальцев
    или кулак с одной стороны и локоть или обрубок у рукава с другой. Какой из них вход,
    решает palm_search_region.
    """
    seeds = np.zeros_like(hand)
    seeds[:2], seeds[-2:], seeds[:, :2], seeds[:, -2:] = hand[:2], hand[-2:], hand[:, :2], hand[:, -2:]
    if seeds.any():
        return [seeds]
    D = cv2.distanceTransform(hand, cv2.DIST_L2, 5)
    _, _, _, p0 = cv2.minMaxLoc(D)
    ends = []
    start = np.zeros_like(hand)
    cv2.circle(start, p0, 3, 255, -1)
    for _ in range(2):
        G = geodesic(hand, start & hand)
        if G is None:
            return []
        _, _, _, far = cv2.minMaxLoc(np.where(hand > 0, G, -1).astype(np.float32))
        start = np.zeros_like(hand)
        cv2.circle(start, far, 6, 255, -1)
        ends.append(start & hand)
    return ends


def geodesic(mask, seeds, scale=4):
    """Геодезическое расстояние (px) от seeds внутри маски — волной дилатаций на уменьшенной маске."""
    h, w = mask.shape
    # INTER_AREA + «> 0»: пиксель — рука, если покрыт хоть частично, иначе тонкие пальцы
    # отрываются от руки при уменьшении и волна до них не доходит
    sm = cv2.resize(mask, (w // scale, h // scale), interpolation=cv2.INTER_AREA) > 0
    sd = cv2.resize(seeds, (w // scale, h // scale), interpolation=cv2.INTER_AREA) > 0
    G = np.full(sm.shape, -1, np.int32)
    cur = (sd & sm).astype(np.uint8)
    if not cur.any():
        return None
    G[cur > 0] = 0
    k = np.ones((3, 3), np.uint8)
    step = 0
    while True:
        step += 1
        nxt = cv2.dilate(cur, k) & sm.astype(np.uint8)
        new = (nxt > 0) & (G < 0)
        if not new.any():
            break
        G[new] = step
        cur = nxt
    G = cv2.resize(G.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) * scale
    G[(mask > 0) & (G < 0)] = G.max()   # мелкие участки, потерянные при уменьшении
    return G


def thickness_profile(hand, seeds, D):
    """Толщина руки (максимум distance transform) по шагам от дальнего от seeds конца."""
    G = geodesic(hand, seeds)
    if G is None:
        return None, None
    inside = hand > 0
    t = ((G.max() - G[inside]) // PROFILE_BIN).astype(int)
    f = np.zeros(t.max() + 1)
    np.maximum.at(f, t, D[inside])
    return np.convolve(f, np.ones(3) / 3, mode="same"), t


def thin_start(f):
    """Сколько шагов от дальнего конца толщина ниже половины максимума (пальцы — долго, локоть — сразу)."""
    return int(np.argmax(f >= 0.5 * f.max()))


def first_dip(f):
    """Индекс провала запястья после первого пика толщины и сам пик (None, если провала нет)."""
    peak = 0.0
    for i, v in enumerate(f):
        if v > peak:
            peak = v
        elif peak >= 0.5 * f.max() and v < PROFILE_DIP * peak:
            return i, peak
    return None, peak


def sleeve_near(seeds, hand, V, bg_v, radius):
    """Доля кольца вокруг точки входа, занятая крупной тёмной областью (рукав)."""
    ys, xs = np.nonzero(seeds)
    ring = np.zeros_like(hand)
    cv2.circle(ring, (int(xs.mean()), int(ys.mean())), max(int(radius), 8), 255, 3)
    ring = (ring > 0) & (hand == 0)
    if not ring.any():
        return 0.0
    dark = ((V < SLEEVE_DARK * bg_v) & (hand == 0)).astype(np.uint8)
    nd, dlab, dst, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    big = [i for i in range(1, nd) if dst[i, cv2.CC_STAT_AREA] >= SLEEVE_MIN_AREA * dark.size]
    return float(np.isin(dlab[ring], big).mean())


def palm_search_region(hand, V, bg_v):
    """(seeds, область поиска ладони) по профилю толщины вдоль руки.

    Если рука касается края кадра, вход — край. Иначе вход — один из двух концов руки,
    выбранный голосованием по трём признакам (голос засчитывается при явной разнице):
      1) противоположный конец начинается тонко (пальцы, а не локоть);
      2) первый «пик + провал» от противоположного конца короткий относительно толщины
         (кисть ~2r, а не предплечье);
      3) у входа рукав (крупная тёмная область).
    Ладонь — первый пик толщины от дальнего конца перед провалом запястья.
    """
    cands = entry_candidates(hand, V, bg_v)
    if not cands:
        return None, None
    D = cv2.distanceTransform(hand, cv2.DIST_L2, 5)
    info = []
    for seeds in cands:
        f, t = thickness_profile(hand, seeds, D)
        if f is None:
            continue
        i, peak = first_dip(f)
        info.append({"seeds": seeds, "f": f, "t": t, "thin": thin_start(f),
                     "ratio": (i * PROFILE_BIN / peak) if i is not None else np.inf,
                     "sleeve": sleeve_near(seeds, hand, V, bg_v, 0.6 * D.max())})
    if not info:
        return None, None
    if len(info) == 1:
        best = info[0]
    else:
        a, b = info
        votes = 0  # > 0 — за a, < 0 — за b
        hi, lo = max(a["thin"], b["thin"]), min(a["thin"], b["thin"])
        if hi >= 10 and hi >= 1.5 * max(lo, 1):
            votes += 1 if a["thin"] > b["thin"] else -1
        if abs(a["ratio"] - b["ratio"]) > 1.0:
            votes += 1 if a["ratio"] < b["ratio"] else -1
        if (a["sleeve"] >= 0.3) != (b["sleeve"] >= 0.3):
            votes += 1 if a["sleeve"] >= 0.3 else -1
        if votes == 0:  # ничья: сначала тонкость, потом длина
            votes = int(np.sign(a["thin"] - b["thin"]))
            if votes == 0 and a["ratio"] != b["ratio"]:
                votes = 1 if a["ratio"] < b["ratio"] else -1
        best = a if votes >= 0 else b
    i, _ = first_dip(best["f"])
    if i is None:
        return best["seeds"], None
    region = np.zeros_like(hand)
    region[hand > 0] = np.where(best["t"] <= i, 255, 0)
    return best["seeds"], region


def touches_border(mask, margin=2):
    return bool(mask[:margin].any() or mask[-margin:].any()
                or mask[:, :margin].any() or mask[:, -margin:].any())


def cut_forearm(hand, V=None, bg_v=None):
    """Удаляет предплечье. Возвращает (маска руки, маска удалённого, центр, r).

    V, bg_v — яркость кадра и фона (для проверки «упирается в рукав»); без них
    удаляются только толстые выступы, уходящие за край кадра.
    """
    seeds, search = palm_search_region(hand, V, bg_v) if V is not None else (None, None)
    rescued = search is not None
    for _ in range(3):
        c, r = palm(hand, search)
        outside = hand.copy()
        cv2.circle(outside, (int(c[0]), int(c[1])), int(CUT_R * r), 0, -1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
        thick_border, thick_free = [], []
        for i in range(1, n):
            part = np.where(lab == i, 255, 0).astype(np.uint8)
            if cv2.distanceTransform(part, cv2.DIST_L2, 5).max() <= FOREARM_THICK * r:
                continue  # тонкий выступ — палец
            if (touches_border(part) or (seeds is not None and np.any((part > 0) & (seeds > 0)))
                    or (V is not None and far_end_dark(part, hand, V, bg_v, c, r))):
                thick_border.append(part)   # уходит за край кадра или в рукав — предплечье
            else:
                thick_free.append(part)
        core = cv2.subtract(hand, outside)
        if touches_border(core) and thick_free and not rescued:
            # центр попал на предплечье у края кадра: кисть — толстый выступ вдали от края
            search = max(thick_free, key=lambda m: int((m > 0).sum()))
            rescued = True
            continue
        removed = np.zeros_like(hand)
        for part in thick_border:
            removed = cv2.bitwise_or(removed, part)
        kept = largest_component(cv2.subtract(hand, removed))
        if kept is None:
            return hand, np.zeros_like(hand), c, r
        region = None
        if search is not None and np.any((search > 0) & (kept > 0)):
            region = np.where((search > 0) & (kept > 0), 255, 0).astype(np.uint8)
        c, r = palm(kept, region)
        return kept, removed, c, r
    return hand, np.zeros_like(hand), c, r


def segment(bgr):
    """Полная сегментация. Возвращает dict или None, если рука не найдена."""
    img = resize_work(bgr)
    skin, thr = skin_mask(img)
    hand = select_hand(skin, img)
    if hand is None:
        return None
    V, bg_v = background_brightness(img, skin)
    hand, removed, c, r = cut_forearm(hand, V, bg_v)
    if hand is None:
        return None
    return {"img": img, "skin": skin, "mask": hand, "removed": removed,
            "center": c, "radius": r, "thr": thr}


def draw(seg):
    """Картинка для отладки: рука зелёная, предплечье красное, отверстия синие, ладонь — круг."""
    vis = (seg["img"] * 0.4).astype(np.uint8)
    vis[seg["mask"] > 0] = (0, 200, 0)
    vis[seg["removed"] > 0] = (0, 0, 220)
    cnts, hier = cv2.findContours(seg["mask"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        for cnt, h in zip(cnts, hier[0]):
            if h[3] >= 0:  # внутренний контур — отверстие
                cv2.drawContours(vis, [cnt], -1, (255, 120, 0), -1)
    c, r = seg["center"], seg["radius"]
    cv2.circle(vis, (int(c[0]), int(c[1])), int(r), (255, 255, 255), 2)
    cv2.circle(vis, (int(c[0]), int(c[1])), int(CUT_R * r), (0, 255, 255), 1)
    return vis
