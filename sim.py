#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""二维小球完全弹性碰撞模拟（纯标准库）。

用法: python3 sim.py <场景.json>
每个固定步长输出一行: t x0 y0 ... x_{N-1} y_{N-1} energy
"""

import json
import sys


def solve(scene):
    width = float(scene["width"])
    height = float(scene["height"])
    dt = float(scene["dt"])
    nsteps = int(round(float(scene["duration"]) / dt))

    xs = [float(b["x"]) for b in scene["balls"]]
    ys = [float(b["y"]) for b in scene["balls"]]
    vxs = [float(b["vx"]) for b in scene["balls"]]
    vys = [float(b["vy"]) for b in scene["balls"]]
    rs = [float(b["radius"]) for b in scene["balls"]]
    ms = [float(b["mass"]) for b in scene["balls"]]
    n = len(xs)

    inv_m = [1.0 / m for m in ms]

    def resolve_walls():
        changed = False
        for i in range(n):
            r = rs[i]
            lo = r
            hi_x = width - r
            if xs[i] < lo:
                xs[i] = lo
                if vxs[i] < 0.0:
                    vxs[i] = -vxs[i]
                changed = True
            elif xs[i] > hi_x:
                xs[i] = hi_x
                if vxs[i] > 0.0:
                    vxs[i] = -vxs[i]
                changed = True
            hi_y = height - r
            if ys[i] < r:
                ys[i] = r
                if vys[i] < 0.0:
                    vys[i] = -vys[i]
                changed = True
            elif ys[i] > hi_y:
                ys[i] = hi_y
                if vys[i] > 0.0:
                    vys[i] = -vys[i]
                changed = True
        return changed

    def resolve_balls():
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                rsum = rs[i] + rs[j]
                d2 = dx * dx + dy * dy
                if d2 >= rsum * rsum:
                    continue
                changed = True
                d = d2 ** 0.5
                if d > 0.0:
                    nx = dx / d
                    ny = dy / d
                else:
                    # 球心完全重合：退化情况，沿 x 轴分开
                    nx, ny = 1.0, 0.0
                    d = 0.0
                # 穿透修复：沿法向各移重叠量的一半
                overlap = rsum - d
                half = 0.5 * overlap
                xs[i] -= nx * half
                ys[i] -= ny * half
                xs[j] += nx * half
                ys[j] += ny * half
                # 完全弹性冲量（仅在两球相互接近时施加）
                rvx = vxs[i] - vxs[j]
                rvy = vys[i] - vys[j]
                vn = rvx * nx + rvy * ny
                if vn > 0.0:
                    jimp = -2.0 * vn / (inv_m[i] + inv_m[j])
                    ix = jimp * nx
                    iy = jimp * ny
                    vxs[i] += ix * inv_m[i]
                    vys[i] += iy * inv_m[i]
                    vxs[j] -= ix * inv_m[j]
                    vys[j] -= iy * inv_m[j]
        return changed

    lines = []
    for step in range(1, nsteps + 1):
        # 固定步长积分
        for i in range(n):
            xs[i] += vxs[i] * dt
            ys[i] += vys[i] * dt
        # 多轮消解：同一步可能既有球-墙又有球-球、三球同时碰撞，
        # 位置修复还可能把球推进墙或另一球里，故反复处理直到干净。
        for _ in range(16):
            w = resolve_walls()
            b = resolve_balls()
            if not w and not b:
                break
        else:
            resolve_walls()  # 最后兜底，保证不出界

        energy = 0.0
        for i in range(n):
            energy += 0.5 * ms[i] * (vxs[i] * vxs[i] + vys[i] * vys[i])

        t = step * dt
        parts = [format(t, ".10g")]
        for i in range(n):
            parts.append(format(xs[i], ".10g"))
            parts.append(format(ys[i], ".10g"))
        parts.append(format(energy, ".12g"))
        lines.append(" ".join(parts))

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python3 sim.py <scene.json>\n")
        return 2
    with open(sys.argv[1], "r") as fh:
        scene = json.load(fh)
    sys.stdout.write(solve(scene))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
