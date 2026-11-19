#!/usr/bin/env python3
"""Съёмка датасета жестов с веб-камеры.

Каждый кадр сохраняется в data/raw/<класс>/ как JPEG, вся разметка — в имени файла:
    <класс>__<вариант>__<рука>__<сторона>__<человек>__<время>.jpg
Например: nozhnitsy__IM__R__palm__andrew__20261119-121530-123.jpg

Управление (окно с видео должно быть активным):
    n / p     следующий / предыдущий класс
    v         следующий вариант жеста внутри класса
    h         рука: правая / левая
    s         сторона: ладонь / тыл
    ПРОБЕЛ    сохранить кадр
    u         удалить последний сохранённый кадр
    q / Esc   выход

Запуск:
    python capture.py --person andrew
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (код для файлов, название, [(код варианта, описание варианта), ...])
CLASSES = [
    ("kamen", "Камень (кулак)", [("-", "кулак")]),
    ("bumaga", "Бумага (ладонь)", [("-", "раскрытая ладонь")]),
    ("nozhnitsy", "Ножницы", [("IM", "указательный + средний"),
                              ("MR", "средний + безымянный"),
                              ("RL", "безымянный + мизинец")]),
    ("klass", "Класс", [("-", "большой палец вверх")]),
    ("ok", "ОК", [("-", "кольцо Б+У, остальные разогнуты")]),
    ("koza", "Коза", [("IL", "указательный + мизинец"),
                      ("TIL", "большой + указательный + мизинец")]),
    ("pozvoni", "Позвони мне", [("-", "большой + мизинец")]),
    ("pistolet", "Пистолет", [("-", "большой + указательный (L)")]),
    ("tri", "Три пальца", [("IMR", "указательный + средний + безымянный"),
                           ("MRL", "средний + безымянный + мизинец")]),
    ("fiksiki", "Фиксики", [("-", "большой + указательный + средний")]),
    ("drugoy", "Другой жест", [("one", "один указательный"),
                               ("four", "четыре пальца"),
                               ("middle", "средний палец"),
                               ("fig", "кукиш"),
                               ("crossed", "скрещённые пальцы"),
                               ("edge", "ладонь ребром"),
                               ("other", "что-то ещё")]),
]
HANDS = [("R", "правая"), ("L", "левая")]
SIDES = [("palm", "ладонь"), ("back", "тыл")]
COMBOS = [(h, s) for h, _ in HANDS for s, _ in SIDES]  # 4 столбца таблицы

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

DISPLAY_H = 540
PANEL_W = 470


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size), path
            except OSError:
                continue
    return ImageFont.load_default(), None


def count_existing(root: Path):
    counts = {}
    for p in root.glob("*/*.jpg"):
        parts = p.stem.split("__")
        if len(parts) >= 4:
            key = tuple(parts[:4])  # класс, вариант, рука, сторона
            counts[key] = counts.get(key, 0) + 1
    return counts


def class_total(counts, cls):
    return sum(n for k, n in counts.items() if k[0] == cls)


def draw_panel(state, counts, fonts, flash):
    f_big, f_mid, f_small = fonts
    cls, cls_name, variants = CLASSES[state["ci"]]
    var_code, var_name = variants[state["vi"]]
    hand_code, hand_name = HANDS[state["hi"]]
    side_code, side_name = SIDES[state["si"]]

    img = Image.new("RGB", (PANEL_W, DISPLAY_H), (32, 34, 38))
    d = ImageDraw.Draw(img)
    y = 10
    d.text((14, y), f"{state['ci'] + 1}/{len(CLASSES)}  {cls_name}", font=f_big,
           fill=(255, 255, 255))
    y += 36
    d.text((14, y), f"Вариант [v]: {var_name}", font=f_mid, fill=(255, 210, 120))
    y += 26
    d.text((14, y), f"Рука [h]: {hand_name}    Сторона [s]: {side_name}", font=f_mid,
           fill=(150, 220, 255))
    y += 34

    # таблица текущего класса: варианты x (рука, сторона)
    col_x = [14, 200, 265, 330, 395]
    heads = ["", "П-лад", "П-тыл", "Л-лад", "Л-тыл"]
    for x, h in zip(col_x, heads):
        d.text((x, y), h, font=f_small, fill=(170, 170, 170))
    y += 20
    for vi, (vc, vn) in enumerate(variants):
        color = (255, 210, 120) if vi == state["vi"] else (220, 220, 220)
        d.text((col_x[0], y), vn[:24], font=f_small, fill=color)
        for ci, (h, s) in enumerate(COMBOS):
            n = counts.get((cls, vc, h, s), 0)
            active = vi == state["vi"] and h == hand_code and s == side_code
            c = (120, 255, 140) if active else ((220, 220, 220) if n else (110, 110, 110))
            d.text((col_x[ci + 1] + 12, y), str(n), font=f_small, fill=c)
        y += 19
    y += 10

    # итоги по всем классам
    d.line((14, y, PANEL_W - 14, y), fill=(70, 70, 70))
    y += 8
    total = 0
    for i, (c, name, _) in enumerate(CLASSES):
        n = class_total(counts, c)
        total += n
        color = (255, 255, 255) if i == state["ci"] else (150, 150, 150)
        d.text((14, y), f"{i + 1:>2}. {name}", font=f_small, fill=color)
        d.text((PANEL_W - 70, y), str(n), font=f_small, fill=color)
        y += 18
    d.text((14, y + 4), f"Всего кадров: {total}", font=f_small, fill=(200, 200, 200))

    d.text((14, DISPLAY_H - 24), "n/p класс  ПРОБЕЛ снять  u отмена  q выход",
           font=f_small, fill=(140, 140, 140))

    panel = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if flash:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, DISPLAY_H - 1), (120, 255, 140), 6)
    return panel


def main():
    ap = argparse.ArgumentParser(description="Съёмка датасета жестов")
    ap.add_argument("--person", default="andrew", help="кто показывает жесты")
    ap.add_argument("--camera", type=int, default=0, help="номер камеры")
    ap.add_argument("--out", default="data/raw", help="куда сохранять")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    root = Path(args.out)
    for c, _, _ in CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)

    f_big, font_path = load_font(26)
    fonts = (f_big, load_font(19)[0], load_font(16)[0])
    print(f"Шрифт для подписей: {font_path or 'встроенный (кириллица может не отображаться)'}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Камера не открылась: проверьте разрешение на камеру для Терминала")

    counts = count_existing(root)
    state = {"ci": 0, "vi": 0, "hi": 0, "si": 0}
    saved_stack = []
    flash_until = 0.0
    print(f"Сохраняю в {root.resolve()}, уже есть кадров: {sum(counts.values())}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Не удалось прочитать кадр")
            break

        # на экране показываем зеркально (как в зеркале), сохраняем исходный кадр
        view = cv2.flip(frame, 1)
        scale = DISPLAY_H / view.shape[0]
        view = cv2.resize(view, (int(view.shape[1] * scale), DISPLAY_H))
        panel = draw_panel(state, counts, fonts, time.time() < flash_until)
        cv2.imshow("capture", np.hstack([view, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("n"):
            state.update(ci=(state["ci"] + 1) % len(CLASSES), vi=0)
        elif key == ord("p"):
            state.update(ci=(state["ci"] - 1) % len(CLASSES), vi=0)
        elif key == ord("v"):
            state["vi"] = (state["vi"] + 1) % len(CLASSES[state["ci"]][2])
        elif key == ord("h"):
            state["hi"] = (state["hi"] + 1) % len(HANDS)
        elif key == ord("s"):
            state["si"] = (state["si"] + 1) % len(SIDES)
        elif key == ord(" "):
            cls = CLASSES[state["ci"]][0]
            var = CLASSES[state["ci"]][2][state["vi"]][0]
            hand = HANDS[state["hi"]][0]
            side = SIDES[state["si"]][0]
            now = time.time()
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
            path = root / cls / f"{cls}__{var}__{hand}__{side}__{args.person}__{ts}.jpg"
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok_enc:
                path.write_bytes(buf.tobytes())
                key4 = (cls, var, hand, side)
                counts[key4] = counts.get(key4, 0) + 1
                saved_stack.append((path, key4))
                flash_until = time.time() + 0.25
                print(f"+ {path.name}")
        elif key == ord("u") and saved_stack:
            path, key4 = saved_stack.pop()
            path.unlink(missing_ok=True)
            counts[key4] -= 1
            print(f"- {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Готово. Всего кадров в датасете: {sum(counts.values())}")


if __name__ == "__main__":
    main()
