#!/usr/bin/env python3
"""Живое демо распознавания жестов с веб-камеры.

    python demo.py                 # камера 0, порог уверенности 0.4
    python demo.py --threshold 0.5   # строже: меньше ответов, меньше ошибок

Слева — кадр: контур руки (зелёный), отрезанное предплечье (красный), ладонь (круг),
ось к запястью (красная стрелка), найденные пальцы (зелёные точки).
Справа — ответ, уверенность и пять самых вероятных классов, внизу нормированная маска
со скелетом, по которой считаются признаки.

Клавиши: q/Esc — выход, m — зеркало вкл/выкл, + / - — порог уверенности,
         s — сохранить снимок экрана и исходный кадр в results/snapshots/
"""
import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw

import features as F
import segment as S
from capture import load_font

VIEW_H = 540
PANEL_W = 430
EMA_ALPHA = 0.4      # вес нового кадра при сглаживании вероятностей
LOST_FRAMES = 8      # столько кадров без руки — сбрасываем сглаживание


class Demo:
    def __init__(self, model_path="model.joblib", threshold=0.4):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1  # один кадр за раз — параллельность только мешает
        self.classes = bundle["classes"]
        self.names = bundle["names"]
        self.threshold = threshold
        self.mirror = True
        self.ema = None
        self.lost = 0
        self.fps = 0.0
        self.fonts = (load_font(34)[0], load_font(20)[0], load_font(16)[0])

    def class_probs(self, vec):
        out = np.zeros(len(self.classes))
        if hasattr(self.model, "predict_proba"):
            pp = self.model.predict_proba(vec[None])[0]
            for j, c in enumerate(self.model.classes_):
                out[self.classes.index(c.split(":")[0])] += pp[j]
        else:
            out[self.classes.index(self.model.predict(vec[None])[0].split(":")[0])] = 1.0
        return out

    def step(self, frame):
        t0 = time.time()
        seg = S.segment(frame)
        dbg = None
        if seg is None:
            self.lost += 1
            if self.lost > LOST_FRAMES:
                self.ema = None
            view = S.resize_work(frame)
        else:
            self.lost = 0
            vec, _, dbg = F.extract(seg, return_debug=True)
            p = self.class_probs(vec)
            self.ema = p if self.ema is None else EMA_ALPHA * p + (1 - EMA_ALPHA) * self.ema
            view = self.draw_view(seg, dbg)
        dt = time.time() - t0
        self.fps = 0.9 * self.fps + 0.1 * (1 / max(dt, 1e-3)) if self.fps else 1 / max(dt, 1e-3)

        if self.mirror:
            view = cv2.flip(view, 1)
        scale = VIEW_H / view.shape[0]
        view = cv2.resize(view, (int(view.shape[1] * scale), VIEW_H))
        return np.hstack([view, self.draw_panel(seg is not None, dbg)])

    def draw_view(self, seg, dbg):
        img = seg["img"].copy()
        red = img.copy()
        red[seg["removed"] > 0] = (0, 0, 220)
        img = cv2.addWeighted(img, 0.6, red, 0.4, 0)
        cnts, _ = cv2.findContours(seg["mask"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 230, 0), 2)
        c = np.array(seg["center"], float)
        r = seg["radius"]
        cv2.circle(img, tuple(int(v) for v in c), int(r), (255, 255, 255), 1)
        u = dbg["u"]
        cv2.arrowedLine(img, tuple(int(v) for v in c), tuple(int(v) for v in c + 1.8 * r * u),
                        (0, 0, 255), 2, tipLength=0.25)
        k = r / F.R0
        for fg in dbg["fingers"]:
            tip = (fg["tip"] - F.CANVAS / 2) * k + c
            cv2.line(img, tuple(int(v) for v in c), tuple(int(v) for v in tip), (0, 255, 0), 1)
            cv2.circle(img, tuple(int(v) for v in tip), 6, (0, 255, 0), -1)
        return img

    def draw_panel(self, hand_found, dbg):
        f_big, f_mid, f_small = self.fonts
        im = Image.new("RGB", (PANEL_W, VIEW_H), (32, 34, 38))
        d = ImageDraw.Draw(im)
        if self.ema is None:
            label, conf, color = ("Рука не найдена" if not hand_found else "…"), 0.0, (170, 170, 170)
        else:
            top = int(np.argmax(self.ema))
            conf = float(self.ema[top])
            if conf >= self.threshold:
                label, color = self.names[self.classes[top]], (120, 255, 140)
            else:
                label, color = "Не уверен", (255, 200, 90)
        d.text((16, 12), label, font=f_big, fill=color)
        d.text((16, 56), f"уверенность {conf:.2f}   порог {self.threshold:.2f} [+/-]",
               font=f_small, fill=(200, 200, 200))
        y = 88
        if self.ema is not None:
            for i in np.argsort(-self.ema)[:5]:
                w = int(190 * self.ema[i])
                d.rectangle((180, y + 3, 180 + w, y + 19), fill=(70, 140, 220))
                d.text((16, y), self.names[self.classes[i]], font=f_mid, fill=(230, 230, 230))
                d.text((180 + w + 6, y), f"{self.ema[i]:.2f}", font=f_small, fill=(200, 200, 200))
                y += 26
        d.text((16, VIEW_H - 24), f"{self.fps:4.1f} кадр/с   m зеркало   s снимок   q выход",
               font=f_small, fill=(140, 140, 140))
        panel = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        if dbg is not None:  # нормированная маска со скелетом и пальцами
            fv = F.draw_features(dbg)[100:380, 100:380]
            fv = cv2.resize(fv, (250, 250))
            panel[VIEW_H - 280:VIEW_H - 30, (PANEL_W - 250) // 2:(PANEL_W + 250) // 2] = fv
        return panel


def main():
    ap = argparse.ArgumentParser(description="Живое демо распознавания жестов")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    demo = Demo(args.model, args.threshold)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Камера не открылась")
    snaps = Path("results/snapshots")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = demo.step(frame)
        cv2.imshow("gestures", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            demo.mirror = not demo.mirror
        elif key in (ord("+"), ord("=")):
            demo.threshold = min(0.95, demo.threshold + 0.05)
        elif key == ord("-"):
            demo.threshold = max(0.05, demo.threshold - 0.05)
        elif key == ord("s"):
            snaps.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(str(snaps / f"{stamp}_screen.png"), out)
            cv2.imwrite(str(snaps / f"{stamp}_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"снимок: {snaps / stamp}_screen.png и _raw.jpg")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
