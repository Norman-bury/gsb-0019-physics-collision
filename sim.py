#!/usr/bin/env python3
import json
import math
import sys


EPS = 1e-12


def solve_ball_collision(balls, i, j):
    a = balls[i]
    b = balls[j]
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    dist = math.hypot(dx, dy)
    min_dist = a["radius"] + b["radius"]

    if dist >= min_dist:
        return False

    if dist <= EPS:
        dx = 1.0
        dy = 0.0
        dist = EPS

    nx = dx / dist
    ny = dy / dist
    overlap = min_dist - dist
    inv_mass_a = 1.0 / a["mass"]
    inv_mass_b = 1.0 / b["mass"]
    inv_total = inv_mass_a + inv_mass_b

    a["x"] -= nx * overlap * (inv_mass_a / inv_total)
    a["y"] -= ny * overlap * (inv_mass_a / inv_total)
    b["x"] += nx * overlap * (inv_mass_b / inv_total)
    b["y"] += ny * overlap * (inv_mass_b / inv_total)

    rvx = a["vx"] - b["vx"]
    rvy = a["vy"] - b["vy"]
    rel_speed = rvx * nx + rvy * ny
    if rel_speed <= 0.0:
        return True

    impulse = -2.0 * rel_speed / inv_total
    a["vx"] += (impulse / a["mass"]) * nx
    a["vy"] += (impulse / a["mass"]) * ny
    b["vx"] -= (impulse / b["mass"]) * nx
    b["vy"] -= (impulse / b["mass"]) * ny
    return True


def constrain_walls(balls, width, height):
    for ball in balls:
        left = ball["radius"]
        right = width - ball["radius"]
        bottom = ball["radius"]
        top = height - ball["radius"]

        if ball["x"] < left:
            ball["x"] = left
            if ball["vx"] < 0.0:
                ball["vx"] = -ball["vx"]
        elif ball["x"] > right:
            ball["x"] = right
            if ball["vx"] > 0.0:
                ball["vx"] = -ball["vx"]

        if ball["y"] < bottom:
            ball["y"] = bottom
            if ball["vy"] < 0.0:
                ball["vy"] = -ball["vy"]
        elif ball["y"] > top:
            ball["y"] = top
            if ball["vy"] > 0.0:
                ball["vy"] = -ball["vy"]


def resolve_ball_collisions(balls):
    count = len(balls)
    for _ in range(max(1, count * count)):
        collisions = []
        for i in range(count):
            for j in range(i + 1, count):
                a = balls[i]
                b = balls[j]
                dx = b["x"] - a["x"]
                dy = b["y"] - a["y"]
                dist = math.hypot(dx, dy)
                min_dist = a["radius"] + b["radius"]
                if dist < min_dist:
                    rvx = a["vx"] - b["vx"]
                    rvy = a["vy"] - b["vy"]
                    if dist <= EPS:
                        nx = 1.0
                        ny = 0.0
                        dist = EPS
                    else:
                        nx = dx / dist
                        ny = dy / dist
                    approach = rvx * nx + rvy * ny
                    collisions.append((min_dist - dist, approach, i, j))

        if not collisions:
            return

        collisions.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        _, _, i, j = collisions[0]
        solve_ball_collision(balls, i, j)


def kinetic_energy(balls):
    return sum(
        0.5 * ball["mass"] * (ball["vx"] ** 2 + ball["vy"] ** 2)
        for ball in balls
    )


def format_number(value):
    return "%.17g" % value


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python3 sim.py <scene.json>\n")
        return 2

    with open(sys.argv[1], "r", encoding="utf-8") as scene_file:
        scene = json.load(scene_file)

    width = float(scene["width"])
    height = float(scene["height"])
    duration = float(scene["duration"])
    dt = float(scene["dt"])
    balls = [
        {
            "x": float(ball["x"]),
            "y": float(ball["y"]),
            "vx": float(ball["vx"]),
            "vy": float(ball["vy"]),
            "radius": float(ball["radius"]),
            "mass": float(ball["mass"]),
        }
        for ball in scene["balls"]
    ]

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not balls:
        raise ValueError("at least one ball is required")
    for ball in balls:
        if ball["mass"] <= 0.0 or ball["radius"] <= 0.0:
            raise ValueError("mass and radius must be positive")
        if not (ball["radius"] <= ball["x"] <= width - ball["radius"]):
            raise ValueError("initial x position is outside bounds")
        if not (ball["radius"] <= ball["y"] <= height - ball["radius"]):
            raise ValueError("initial y position is outside bounds")

    steps = int(round(duration / dt))
    lines = []

    def emit(step):
        values = [format_number(step * dt)]
        for ball in balls:
            values.extend((format_number(ball["x"]), format_number(ball["y"])))
        values.append(format_number(kinetic_energy(balls)))
        lines.append(" ".join(values))

    emit(0)
    for step in range(1, steps + 1):
        for ball in balls:
            ball["x"] += ball["vx"] * dt
            ball["y"] += ball["vy"] * dt

        constrain_walls(balls, width, height)
        resolve_ball_collisions(balls)
        constrain_walls(balls, width, height)
        resolve_ball_collisions(balls)
        emit(step)

    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
