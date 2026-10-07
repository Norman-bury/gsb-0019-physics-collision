#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task19（二维小球碰撞模拟）判分脚本。

规则：
1. 交付 sim.py，命令行入口 `python3 sim.py <场景.json>`。
   场景 JSON 含 width/height/duration/dt 和一个 balls 数组（x,y,vx,vy,radius,mass）。
   sim.py 用固定步长 dt 模拟到 duration，每步输出一行：
       t x0 y0 x1 y1 ... x_{N-1} y_{N-1} energy
   位置按场景 balls 数组顺序，energy 是总动能。
2. 判分脚本内置场景生成（固定种子），跑 sim.py，检查：
   - 任何时刻不穿透：球在边界内、球两两不重叠。
   - 能量守恒：每帧总动能相对初值漂移小于 1%。
   - 球真的在动：总位移大于阈值，防止输出冻结假轨迹。
   - 确定性：相同场景跑两遍输出完全一致。

用法：
  python3 grade.py
  python3 grade.py --sim path/to/sim.py
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

TIMEOUT = 60          # 单个场景限时，秒
ENERGY_TOL = 0.01     # 能量相对漂移阈值 1%
PEN_TOL = 1e-6        # 穿透判定容差
MOVE_TOL = 0.1        # 总位移下限


def gen_scenarios():
    rng = random.Random(20261019)
    scenarios = []
    widths = [10.0, 10.0, 12.0, 10.0, 8.0]
    heights = [10.0, 8.0, 10.0, 12.0, 10.0]
    counts = [2, 3, 4, 3, 2]
    for W, H, N in zip(widths, heights, counts):
        balls = []
        for _ in range(N):
            r = rng.uniform(0.3, 0.6)
            m = rng.uniform(0.5, 2.0)
            # 拒绝采样：球在边界内，且彼此不重叠
            for _ in range(10000):
                x = rng.uniform(r + 0.05, W - r - 0.05)
                y = rng.uniform(r + 0.05, H - r - 0.05)
                ok = True
                for b in balls:
                    if math.hypot(x - b["x"], y - b["y"]) < r + b["radius"] + 0.05:
                        ok = False
                        break
                if ok:
                    break
            vx = rng.uniform(-3.0, 3.0)
            vy = rng.uniform(-3.0, 3.0)
            balls.append({"x": x, "y": y, "vx": vx, "vy": vy,
                          "radius": r, "mass": m})
        # 保证初始动能非零（至少有一个球明显在动）
        if max(math.hypot(b["vx"], b["vy"]) for b in balls) < 0.5:
            balls[0]["vx"] = 1.5
            balls[0]["vy"] = -1.0
        scenarios.append({
            "width": W, "height": H, "duration": 10.0, "dt": 0.001,
            "balls": balls,
        })
    return scenarios


def initial_energy(sc):
    return sum(0.5 * b["mass"] * (b["vx"] ** 2 + b["vy"] ** 2) for b in sc["balls"])


def run_sim(sim_path, scenario):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="scene_", dir=HERE)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(scenario, fh)
        try:
            r = subprocess.run([sys.executable, sim_path, path],
                               cwd=HERE, capture_output=True, text=True,
                               timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return None, "超时"
    finally:
        os.unlink(path)
    if r.returncode != 0:
        return None, r.stderr[-200:]
    return r.stdout, ""


def parse_frames(stdout, N):
    frames = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue  # 跳过无法解析的行（如注释/表头）
        if len(nums) != 1 + 2 * N + 1:
            continue
        t = nums[0]
        pos = [(nums[1 + 2 * i], nums[2 + 2 * i]) for i in range(N)]
        E = nums[-1]
        frames.append((t, pos, E))
    return frames


def check_scenario(sc, frames):
    N = len(sc["balls"])
    W, H = sc["width"], sc["height"]
    dt = sc["dt"]
    E0 = initial_energy(sc)
    init_pos = [(b["x"], b["y"]) for b in sc["balls"]]
    radii = [b["radius"] for b in sc["balls"]]

    # 帧数应接近 duration/dt，末帧时间接近 duration
    expect = round(sc["duration"] / dt)
    if abs(len(frames) - expect) > 2:
        return False, "帧数 %d 与期望 %d 差太多" % (len(frames), expect)
    if frames and abs(frames[-1][0] - sc["duration"]) > 2 * dt:
        return False, "末帧时间 %.6f 没跑到 duration" % frames[-1][0]

    max_edrift = 0.0
    max_pen = 0.0
    for t, pos, E in frames:
        # 能量守恒
        drift = abs(E - E0) / E0 if E0 > 0 else abs(E - E0)
        if drift > max_edrift:
            max_edrift = drift
        # 穿透
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            if x < r - PEN_TOL or x > W - r + PEN_TOL or y < r - PEN_TOL or y > H - r + PEN_TOL:
                return False, "t=%.4f 球%d 越界 (%.6f,%.6f)" % (t, i, x, y)
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi = pos[i]
                xj, yj = pos[j]
                d = math.hypot(xi - xj, yi - yj)
                if d < radii[i] + radii[j] - PEN_TOL:
                    return False, "t=%.4f 球%d 球%d 重叠 d=%.6f" % (t, i, j, d)

    if max_edrift > ENERGY_TOL:
        return False, "能量最大漂移 %.4f 超过 1%%" % max_edrift

    # 球真的在动
    if frames:
        fin_pos = frames[-1][1]
        move = sum(math.hypot(fin_pos[i][0] - init_pos[i][0],
                              fin_pos[i][1] - init_pos[i][1]) for i in range(N))
        if move < MOVE_TOL:
            return False, "总位移 %.4f 太小，疑似冻结" % move

    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(HERE, "sim.py"))
    args = ap.parse_args()

    print("=" * 60)
    print("判分开始")

    if not os.path.exists(args.sim):
        print("[FAIL] 找不到 sim.py（%s）" % args.sim)
        sys.exit(1)

    scenarios = gen_scenarios()
    fails = 0
    for idx, sc in enumerate(scenarios):
        N = len(sc["balls"])
        out1, err = run_sim(args.sim, sc)
        if out1 is None:
            print("[FAIL] 场景%d 运行失败：%s" % (idx, err[-120:]))
            fails += 1
            continue
        frames = parse_frames(out1, N)
        if not frames:
            print("[FAIL] 场景%d 没解析到有效帧" % idx)
            fails += 1
            continue
        ok, msg = check_scenario(sc, frames)
        if not ok:
            print("[FAIL] 场景%d（%d球）：%s" % (idx, N, msg))
            fails += 1
            continue
        # 确定性：再跑一遍，输出必须完全一致
        out2, err2 = run_sim(args.sim, sc)
        if out2 is None:
            print("[FAIL] 场景%d 复跑失败：%s" % (idx, err2[-120:]))
            fails += 1
            continue
        if out1 != out2:
            print("[FAIL] 场景%d 两次运行输出不一致（非确定）" % idx)
            fails += 1
            continue
        print("[OK] 场景%d（%d球，%d帧）" % (idx, N, len(frames)))

    print("=" * 60)
    print("共 %d 个场景，错 %d 个" % (len(scenarios), fails))
    ok = (fails == 0)
    print("总结论：", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
