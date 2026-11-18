#!/usr/bin/env python3
"""4단계(개선): 로봇 손끝을 '기준 바늘'로 쓰는 카메라 보정 수집.

  python steps/s4b_collect_pointer.py --index 0              # 계획만 (모터 이동 없음)
  python steps/s4b_collect_pointer.py --index 0 --execute    # 실제 수집

점마다:
  1) 로봇이 그리퍼를 닫은 채, 손끝이 '블록 윗면 높이 + 3 mm'가 되도록 점 위에 멈춤
  2) 사용자가 빨간 블록을 손끝 바로 아래에 놓음 (블록 윗면 '중앙'이 손끝 바로 밑)
  3) 로봇이 위로 빠져 옆으로 비킴 → 카메라 측정 (카메라 흔들림은 배경 특징점으로 자동 보정)
  → 블록은 그대로 두고 다음 점으로 (다음 점에서 다시 옮겨 놓음)

이전 방식(로봇이 블록을 쥐고 놓기)은 블록이 그리퍼 안에서 매번 다른 위치에 잡혀(최대 ~1 cm)
보정 데이터가 흔들렸음 (11/16 수집: 검증 오차 평균 7 mm). 손끝 위치는 1단계에서 ~1 mm로 검증됨.

결과: steps/logs/s4b_<시각>.jsonl, 기준 화면 steps/logs/s4b_<시각>_reference.png
다음: python steps/s4_solve_camera.py steps/logs/s4b_<시각>.jsonl --save
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.arm import ARM, Arm, ArmStop  # noqa: E402
from robot.camera import make_camera  # noqa: E402
from robot.kinematics import SO101Kinematics  # noqa: E402
from robot.motion import go, plan_arc, plan_line  # noqa: E402
from robot import workcell as W  # noqa: E402

DEFAULT_POINTS = "15,-7 15,0 15,7 19,-7 19,0 19,7 23,-5 23,0 23,5 17,3 21,-3 17,-4 21,4 16,-2 22,2"
# 11/17: 왼쪽(y>+10)·가까운 쪽에서 집기 실패 → 실제로 블록을 둘 범위 전체를 덮는 점 (팬 축 기준 오른쪽→왼쪽 순서)
WIDE_POINTS = ("13,-12 17,-12 21,-10 24,-6 13,-6 17,-7 21,-4 24,0 15,0 19,-3 17,0 19,3 21,4 24,6 "
               "13,6 17,7 21,10 10,10 13,12 17,12 21,15 10,15 13,17 17,17")
HOVER_ABOVE_TOP_M = 0.003          # 손끝이 블록 윗면보다 3 mm 위
PARK_XY = (0.10, -0.14)
PARK_ABOVE_TABLE_M = 0.10
START_MARGIN_DEG = 1.0     # 11/17: 접힌 휴식 자세(팔꿈치 ~90°)에서도 시작 가능하게, 범위 밖만 거부
MAX_SPREAD_PX = 3.0
WARN_DRIFT_PX = 40.0               # 이보다 크게 밀렸으면 카메라 거치대 확인 권고 (보정은 됨)
COLOR = "red"


def ask(prompt):
    return input(prompt).strip().lower()


def parse_points(s):
    return [tuple(float(v) / 100 for v in tok.split(",")) for tok in s.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=DEFAULT_POINTS, help='cm "x,y x,y ..."')
    ap.add_argument("--wide", action="store_true", help="넓은 범위 24점 (왼쪽·가까운 쪽 포함)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--index", default="0", help="위 카메라 번호 또는 /dev/v4l/by-path/... 경로")
    ap.add_argument("--resume", help="중간에 멈춘 수집 로그(.jsonl)에 이어서 — 같은 기준 화면을 사용, 측정된 점은 건너뜀")
    args = ap.parse_args()

    kin = SO101Kinematics()                 # 손끝 접촉점 기준 (1단계 실측)
    pts = parse_points(WIDE_POINTS if args.wide else args.points)
    top = W.TABLE_Z_M + W.BLOCK_M
    z_hover, z_travel = top + HOVER_ABOVE_TOP_M, W.TABLE_Z_M + W.TRAVEL_ABOVE_TABLE_M
    z_park = W.TABLE_Z_M + PARK_ABOVE_TABLE_M

    arm = Arm(port=args.port).connect()
    limits = {m: arm.range_deg(m) for m in ARM}
    cam, ref = None, None
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log, log_path = [], ROOT / f"steps/logs/s4b_{stamp}.jsonl"
    done = set()
    if args.resume:
        log_path = Path(args.resume).resolve()
        log = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        log = [r for r in log if r.get("camera")]
        done = {round(r["target_xy_m"][0] * 100, 1) * 1000 + round(r["target_xy_m"][1] * 100, 1) for r in log}
        print(f"[이어서] {log_path.name}: 측정된 {len(log)}점은 건너뜀")
    try:
        print(f"[기준] 책상 z={W.TABLE_Z_M*1000:.1f} mm, 손끝 멈춤 높이 = 블록 윗면 + {HOVER_ABOVE_TOP_M*1000:.0f} mm")
        q_ref = None
        for i, (x, y) in enumerate(pts):
            ok, r = W.reach_ok(x, y)
            if not ok:
                raise ArmStop(f"({x*100:.0f},{y*100:.0f}) cm: 거리 {r*100:.1f} cm가 허용 범위 밖")
            for z in (z_travel, z_hover):
                res = kin.ik_topdown(x, y, z, limits=limits, q_ref=q_ref)
                if not res.ok:
                    raise ArmStop(f"({x*100:.0f},{y*100:.0f}) z={z*1000:.0f}mm: {res.problems}")
                q_ref = res.q
            print(f"  점{i+1:2d} ({x*100:5.1f},{y*100:5.1f}) cm → " + " ".join(f"{v:6.1f}" for v in res.q.values()))
        if not kin.ik_topdown(*PARK_XY, z_park, limits=limits).ok:
            raise ArmStop("주차 자세 IK 실패")

        cam = make_camera("top", args.index).start()
        if args.resume:
            from robot.camera import ViewReference
            ref = ViewReference.load(log_path.with_name(f"{log_path.stem}_reference.png"))
            M, n = ref.register(cam.frame()[0])
            if M is None:
                raise ArmStop("이전 기준 화면과 지금 화면이 맞지 않음 — 처음부터 다시 수집하세요")
            print(f"[이어서] 기준 화면과 비교: 카메라 밀림 {ref.shift_px(M):.1f}px (특징점 {n}개) — 자동 보정됨")
        m, diag = cam.measure(COLOR, n_frames=10)
        print(f"[카메라] '{COLOR}' 검출: {diag['detections_per_frame']}"
              + (f", 픽셀 {np.round(m['pixel'], 0)}" if m else "") + "  (블록 1개만 화면에 있어야 함)")

        now = arm.read_deg()
        bad = [f"{j}={now[j]:.1f}°" for j in ("shoulder_lift", "elbow_flex", "wrist_flex")
               if not (arm.range_deg(j)[0] + START_MARGIN_DEG < now[j] < arm.range_deg(j)[1] - START_MARGIN_DEG)]
        if bad:
            print(f"[거부] 시작 자세가 측정 범위 밖: {bad}. 토크 OFF 상태에서 팔을 범위 안으로 옮긴 뒤 다시.")
            return
        if not args.execute:
            print("[계획만] 실제 수집은 --execute")
            return

        ask("\n팔을 지금 자세 그대로 받치고 Enter (토크 ON) > ")
        start_q = arm.read_deg()
        arm.torque_on()
        arm.gripper_torque_on()
        time.sleep(0.3)
        ask("자세가 유지되면 손을 떼고 Enter (그리퍼를 닫음 — 손가락 사이에 아무것도 없게) > ")
        arm.gripper_move_raw(arm._gripper_raw(0))

        cur, q_ref = None, None
        for i, (x, y) in enumerate(pts):
            if round(x * 100, 1) * 1000 + round(y * 100, 1) in done:
                continue
            above, hover = np.array([x, y, z_travel]), np.array([x, y, z_hover])
            print(f"\n[점 {i+1}/{len(pts)}] ({x*100:.1f}, {y*100:.1f}) cm")
            if cur is None:
                r0 = kin.ik_topdown(*above, limits=limits)
                res = arm.move_deg(r0.q, result_tol=40)
                if not res.status.startswith("도착"):
                    raise ArmStop(res.status)
                q_ref = r0.q
            else:
                wps, q_ref = plan_arc(kin, limits, cur, above, q_ref)
                go(arm, wps, precise=False)
            if cur is not None or args.resume:
                ask("  블록을 책상에서 집어 들고 Enter (로봇이 내려감 — 이전 블록에 부딪히지 않게) > ")
            wps, q_ref = plan_line(kin, limits, above, hover, q_ref)
            go(arm, wps)
            q_act = arm.read_deg()
            tip, _ = kin.fk([q_act[j] for j in ARM])
            ask("  빨간 블록을 손끝 바로 아래에 놓고 Enter\n"
                "   (손끝이 블록 윗면 '중앙' 바로 위 — 위에서 내려다보며 맞추기 / 블록 방향은 상관없음) > ")
            ask("  손을 카메라 화면 밖으로 치우고 Enter (로봇이 위로 빠져 옆으로 비킴) > ")

            wps, q_ref = plan_line(kin, limits, hover, above, q_ref)
            go(arm, wps, precise=False)
            pk = np.array([*PARK_XY, z_park])
            wps, q_park = plan_arc(kin, limits, above, pk, q_ref)
            go(arm, wps, precise=False)
            time.sleep(0.5)
            if ref is None:
                ref = cam.reference_view()
                ref.save(log_path.with_name(f"{log_path.stem}_reference.png"))
            while True:
                m, diag = cam.measure(COLOR, n_frames=15, ref=ref,
                                      save_image=log_path.with_name(f"{log_path.stem}_p{i+1}.png"))
                if m is not None and diag.get("spread_px", 0) <= MAX_SPREAD_PX:
                    break
                a = ask(f"  [카메라] 측정 실패/흔들림 {diag}. 손·다른 빨간 물체 확인 후 Enter=재측정, s=이 점 건너뜀 > ")
                if a == "s":
                    m = None
                    break
            if m:
                drift = diag.get("drift_px", 0.0)
                print(f"  카메라: 픽셀 {np.round(m['pixel'], 1)} (흔들림 {diag['spread_px']:.1f}px, "
                      f"카메라 밀림 보정 {drift:.1f}px)")
                if drift > WARN_DRIFT_PX:
                    print(f"  [주의] 카메라가 처음보다 {drift:.0f}px 밀렸습니다. 보정은 되지만 거치대가 흔들리는지 확인하세요.")
            log = [r for r in log if r["index"] != i + 1]
            log.append({"index": i + 1, "mode": "pointer", "target_xy_m": [x, y], "roll_off_deg": 0.0,
                        "q_hover_deg": q_act, "tip_xy_m": tip[:2].tolist(), "tip_z_m": float(tip[2]),
                        "camera": m, "camera_diag": diag, "camera_kind": "top",
                        "table_z_m": W.TABLE_Z_M, "time": time.time()})
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), encoding="utf-8")
            wps, q_ref = plan_arc(kin, limits, pk, above, q_park)
            go(arm, wps, precise=False)
            cur = above

        print(f"\n[완료] {sum(1 for r in log if r['camera'])}점 수집 → {log_path}")
        res = arm.move_deg({j: start_q[j] for j in ARM}, result_tol=40)
        print(f"[복귀] {res.summary()}")
    except ArmStop as e:
        print(f"\n[STOP] {e} — 현재 위치에서 정지(유지 중)")
    finally:
        if cam is not None:
            cam.stop()
        if log:
            print(f"[LOG] {log_path}\n  다음: python steps/s4_solve_camera.py {log_path.relative_to(ROOT)} --save")
        try:
            torque_on = any(arm._read("Torque_Enable").values())
        except Exception as e:
            print(f"[WARN] 토크 상태를 읽지 못함({e})")
            return
        if torque_on and not any(arm.torque_was.values()):
            try:
                ask("\n팔을 받치고 Enter → 토크 OFF > ")
            except (KeyboardInterrupt, EOFError):
                print("\n[주의] 입력 없이 종료 — 토크를 끕니다")
            arm.close(disable_torque=True)
        else:
            arm.close(disable_torque=False)


if __name__ == "__main__":
    main()
