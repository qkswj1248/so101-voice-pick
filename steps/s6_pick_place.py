"""블록을 찾아 집고 → 컨베이어(가르친 자세)에 놓고 → 시작 자세로 복귀. 반복.

먼저 한 번: python steps/s6_teach_drop.py   (놓을 자세 가르치기)
실행:       python steps/s6_pick_place.py --index 0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.arm import ARM, Arm, ArmStop  # noqa: E402
from robot.kinematics import SO101Kinematics  # noqa: E402
from robot.motion import go  # noqa: E402
from robot.perception import BlockLocator  # noqa: E402
from robot import workcell as W  # noqa: E402
from robot.pick import (PARK_XY, PARK_ABOVE_TABLE_M, PICK_CLEAR_M, OPEN_PICK_PCT, OPEN_RELEASE_PCT,  # noqa: E402
                          ROLL_CHUNK_DEG, START_MARGIN_DEG, pick_yaw, line, travel)

DROP_FILE = ROOT / "calibration/drop_conveyor.json"
COLOR_KO = {"red": "빨간", "orange": "주황", "yellow": "노란", "green": "초록", "blue": "파란", "purple": "보라"}
COLOR_WORDS = {   # voice/voice_test.py와 같은 단어 + 영어 (직접 입력용)
    "red": ["빨간", "빨강", "레드", "red"], "blue": ["파란", "파랑", "블루", "blue"],
    "yellow": ["노란", "노랑", "옐로", "yellow"], "green": ["초록", "녹색", "그린", "green"],
    "orange": ["주황", "오렌지", "orange"], "purple": ["보라", "퍼플", "purple"],
}


def parse_color(text):
    t = text.strip().lower()
    for c, words in COLOR_WORDS.items():
        if any(w in t for w in words):
            return c
    return None


def listen_color():
    """voice/voice_test.py의 recognize_voice() (4초 녹음 → 구글 음성인식 → 색)."""
    try:
        from voice.voice_test import recognize_voice
    except ImportError as e:
        print(f"  [음성] 모듈을 못 불러옴: {e}\n"
              "         이 환경(conda lerobot)에 설치: pip install SpeechRecognition sounddevice\n"
              "         지금은 색 이름을 직접 입력해도 됩니다 (예: 빨강, blue)")
        return None
    return recognize_voice()
HOME_FILE = ROOT / "calibration/home_pose.json"
# 기본 복귀(초기) 자세 = 11/17 접힌 휴식 자세 실측(raw). --set-home으로 바꿀 수 있음
DEFAULT_HOME_RAW = {"shoulder_pan": 2081, "shoulder_lift": 1080, "elbow_flex": 2790, "wrist_flex": 2962, "wrist_roll": 2048}
CARRY_ABOVE_TABLE_M = (0.16, 0.14, 0.12, 0.10)   # 블록을 든 채 옮길 높이 (닿는 가장 높은 값)
DROP_RAISE_M = 0.01     # 11/17: 가르친 놓기 자세가 벨트에 너무 닿음 → 이만큼 위에서 놓음 (--drop-raise-cm)
CLOSED_PCT = 3          # 복귀할 때 그리퍼 (입 닫기)


def ask(p):
    return input(p).strip().lower()


def unwind_roll(arm):
    """손목이 많이 돌아가 있으면 조금씩 0°로 (큰 회전 한 번은 느려짐, 11/17)."""
    roll = arm.read_deg()["wrist_roll"]
    if abs(roll) > ROLL_CHUNK_DEG:
        n = int(math.ceil(abs(roll) / ROLL_CHUNK_DEG))
        for k in range(1, n + 1):
            res = arm.move_deg({"wrist_roll": roll * (1 - k / n)}, correct=(k == n), result_tol=40)
            if res.status.startswith("정지"):
                raise ArmStop("손목 되돌리기 " + res.status)


def raise_pose(kin, q, dz, limits):
    """관절 자세 q를 dz만큼 위로 (어깨·팔꿈치 2관절, 수치 IK). 손목은 그대로 — 가르친 자세가 손목 한계(95°/101.8°)
    근처라 그리퍼 방향까지 유지하면 손목이 범위를 넘음(11/17) → 그리퍼가 몇 도 기울어지는 건 허용."""
    J3 = ("shoulder_lift", "elbow_flex")
    qv = lambda q_: [q_[j] for j in ARM]  # noqa: E731
    p0, R0 = kin.fk(qv(q))
    target, ax0 = p0 + np.array([0, 0, dz]), R0[:, 2]

    def resid(q_):
        p, R = kin.fk(qv(q_))
        return p - target
    q = dict(q)
    for _ in range(40):
        r = resid(q)
        if np.linalg.norm(r[:3]) < 2e-4:
            break
        J = np.zeros((3, len(J3)))
        for i, j in enumerate(J3):
            qq = dict(q); qq[j] += 0.05
            J[:, i] = (resid(qq) - r) / 0.05
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        step *= min(1.0, 3.0 / max(1e-9, np.abs(step).max()))   # 방향은 유지하고 크기만 제한
        for i, j in enumerate(J3):
            q[j] += float(step[i])
    p, _ = kin.fk(qv(q))
    if abs(p[2] - target[2]) > 0.002 or any(not (limits[j][0] < q[j] < limits[j][1]) for j in J3):
        raise ArmStop(f"놓기 자세를 {dz*100:.1f}cm 올릴 수 없음 (팔 범위)")
    return q


def joint_move(arm, q, what, precise=False):
    res = arm.move_deg({j: q[j] for j in ARM}, correct=precise, result_tol=None if precise else 40)
    if not res.status.startswith("도착"):
        raise ArmStop(f"{what}: {res.status}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="0")
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--color", default="red", help="--voice 없이 쓸 때 집을 색")
    ap.add_argument("--voice", action="store_true", help="음성으로 색 선택 (Enter 누르고 말하기)")
    ap.add_argument("--clear-mm", type=float, default=PICK_CLEAR_M * 1000)
    ap.add_argument("--open-pct", type=float, default=OPEN_PICK_PCT)
    ap.add_argument("--drop-raise-cm", type=float, default=DROP_RAISE_M * 100, help="가르친 놓기 자세보다 위에서 놓기(cm)")
    ap.add_argument("--set-home", action="store_true", help="지금 자세(토크 OFF로 옮겨 둔)를 복귀 자세로 저장하고 끝")
    args = ap.parse_args()

    if not DROP_FILE.is_file():
        print(f"[없음] {DROP_FILE} — 먼저 python steps/s6_teach_drop.py 로 놓을 자세를 가르쳐 주세요")
        return
    drop = json.loads(DROP_FILE.read_text(encoding="utf-8"))
    q_app, q_drop = drop["approach"]["q_deg"], drop["drop"]["q_deg"]
    # 11/17: 접근·놓기 자세가 같게 저장돼 있었음(놓은 뒤 안 올라가고 그 자리에서 그리퍼를 닫아 블록을 다시 집음)
    # → 접근 자세가 놓기보다 3cm 이상 높을 때만 쓰고, 아니면 '들고 온 높이' 자세로 빠짐
    use_app = drop["approach"]["block_center_m"][2] - drop["drop"]["block_center_m"][2] >= 0.03
    if not use_app:
        print("[알림] 가르친 접근 자세가 놓기 자세보다 3cm 이상 높지 않음 → 들고 온 높이로 접근·후퇴")

    G = W.grasp_point()
    kin_g = SO101Kinematics(contact=G)
    kin_pick = SO101Kinematics(contact=G + np.array([-args.clear_mm / 1000, 0, 0]))
    kin_tip = SO101Kinematics()
    z_c = W.block_center_z()
    z_travel = W.TABLE_Z_M + W.TRAVEL_ABOVE_TABLE_M
    z_park = W.TABLE_Z_M + PARK_ABOVE_TABLE_M

    arm = Arm(port=args.port).connect()
    limits = {m: arm.range_deg(m) for m in ARM}
    if args.set_home:
        q = arm.read_deg()
        prob = arm.check_targets_deg({j: q[j] for j in ARM})
        if prob:
            print(f"[거부] 범위 끝 5° 안: {prob}")
        else:
            HOME_FILE.write_text(json.dumps({"q_deg": {j: q[j] for j in ARM}}, indent=2), encoding="utf-8")
            print(f"[저장] 복귀 자세 {HOME_FILE}")
        arm.close(disable_torque=False)
        return
    if HOME_FILE.is_file():
        home_q = json.loads(HOME_FILE.read_text(encoding="utf-8"))["q_deg"]
    else:
        home_q = {j: arm.raw_to_deg(j, DEFAULT_HOME_RAW[j]) for j in ARM}
    if args.drop_raise_cm:
        q_drop = raise_pose(kin_g, q_drop, args.drop_raise_cm / 100, limits)
        print(f"[놓기] 가르친 자세보다 {args.drop_raise_cm:.1f} cm 위에서 놓음")
    loc = BlockLocator(index=args.index).start()
    log_path = ROOT / f"steps/logs/s6_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = []
    try:
        for name, q in (("접근", q_app), ("놓기", q_drop), ("복귀", home_q)):
            prob = arm.check_targets_deg(q)
            if prob:
                raise ArmStop(f"가르친 {name} 자세가 범위 끝: {prob} — s6_teach_drop.py 다시")
        r_park = kin_tip.ik_topdown(*PARK_XY, z_park, limits=limits)
        if not r_park.ok:
            raise ArmStop("주차 자세 IK 실패")
        now = arm.read_deg()
        bad = [f"{j}={now[j]:.1f}°" for j in ("shoulder_lift", "elbow_flex", "wrist_flex")
               if not (limits[j][0] + START_MARGIN_DEG < now[j] < limits[j][1] - START_MARGIN_DEG)]
        if bad:
            print(f"[거부] 시작 자세가 측정 범위 밖: {bad}")
            return

        ask("\n시작하려면 Enter (토크 ON → 초기 자세로 이동) > ")
        arm.torque_on()
        arm.gripper_torque_on()
        time.sleep(0.3)
        unwind_roll(arm)
        arm.gripper_open(CLOSED_PCT)
        joint_move(arm, home_q, "초기 자세로")

        cycle = 0
        while True:
            if args.voice:
                a = ask(f"\n[{cycle+1}] Enter 누르고 4초 안에 말하기 (예: '빨간 블록 가져와') / 색 이름 직접 입력 / q = 끝 > ")
                if a == "q":
                    break
                color = parse_color(a) if a else listen_color()
                if color is None:
                    print("  [음성] 색을 못 알아들었어요 — 다시")
                    continue
                print(f"  → {COLOR_KO[color]} 블록 ({color})")
            else:
                a = ask(f"\n[{cycle+1}] 블록을 놓고 손을 치운 뒤 Enter (q = 끝내기) > ")
                if a == "q":
                    break
                color = args.color
            cycle += 1
            t0 = time.monotonic()
            b, diag = loc.find(color)
            if b is None:
                cnt = diag.get("detections_per_frame", {})
                if not diag.get("error") and cnt and max(cnt, key=cnt.get) == 0:
                    print(f"  [카메라] 작업 영역에 {COLOR_KO[color]} 블록이 없음")
                elif not diag.get("error") and cnt and isinstance(max(cnt, key=cnt.get), int) and max(cnt, key=cnt.get) > 1:
                    print(f"  [카메라] 작업 영역에 {COLOR_KO[color]} 블록이 여러 개 — 하나만 두세요")
                else:
                    print(f"  [카메라] 블록을 못 찾음: {diag.get('error') or cnt}")
                continue
            if diag.get("warning"):
                print(f"  [카메라 경고] {diag['warning']}")
            x, y, _ = b["xyz_m"]
            ok, r = W.reach_ok(x, y)
            print(f"  블록: ({x*100:.1f}, {y*100:.1f}) cm, 방향 {b['yaw_deg']:+.1f}°")
            if not ok:
                print(f"  [건너뜀] 거리 {r*100:.1f} cm — 작업 범위 밖")
                continue
            sel = pick_yaw(kin_g, (x, y, z_c), b["yaw_deg"], limits, r_park.q)
            if sel is None:
                print("  [건너뜀] 이 위치·방향으로 집는 자세가 없음")
                continue
            yaw = sel[0]
            above = np.array([x, y, z_travel])
            r_above = kin_g.ik_topdown(*above, yaw_deg=yaw, limits=limits, q_ref=sel[1].q, allow_flip=False)
            if not r_above.ok:
                print(f"  [건너뜀] 블록 위 자세 IK 실패: {r_above.problems}")
                continue

            T = {}
            arm.trace, arm.trace_t0 = [], time.monotonic()
            arm.trace_label = "블록 위로"
            def lap(name, nxt=None, t=[time.monotonic()]):
                now_ = time.monotonic(); T[name] = round(now_ - t[0], 1); t[0] = now_
                arm.trace_label = nxt or ""

            print("  - 집으러 이동")
            joint_move(arm, r_park.q, "대기 자세로")
            wps, q_ref = travel(kin_g, limits, np.array([*PARK_XY, z_park]), above, r_park.q, r_above.q)
            go(arm, wps, precise=False)
            lap("블록 위로", "내려가 집기")
            arm.gripper_open(args.open_pct)
            wps, q_ref = line(kin_pick, limits, above, (x, y, z_c), q_ref, yaw)
            go(arm, wps)
            held, pct = arm.grip()
            lap("내려가 집기", "들어 올리기")
            rec = {"cycle": cycle, "color": color, "xy_m": [x, y], "yaw_deg": b["yaw_deg"], "held": held, "gripper_pct": pct}
            # 11/17: 블록 높이(8cm)에서 바로 관절 이동하면 바닥을 쓸며 감 → 수직으로 높이 든 뒤, 팬만 돌려 컨베이어 쪽으로
            z_carry = None
            for h in CARRY_ABOVE_TABLE_M:
                z_try = W.TABLE_Z_M + h
                if kin_g.ik_topdown(x, y, z_try, yaw_deg=yaw, limits=limits, q_ref=q_ref, allow_flip=False).ok:
                    z_carry = z_try
                    break
            top = np.array([x, y, z_carry if (held and z_carry) else z_travel])
            wps, q_ref = line(kin_g, limits, (x, y, z_c), top, q_ref, yaw)
            go(arm, wps, precise=False)
            lap("들어 올리기", "컨베이어에 놓기")
            if not held:
                print(f"  - 집기 실패 (그리퍼 {pct:.0f}%) → 복귀")
            else:
                print(f"  - 집음 (그리퍼 {pct:.0f}%), {(top[2]-W.TABLE_Z_M)*100:.0f}cm 높이로 → 컨베이어로")
                q_turn = dict(arm.read_deg())
                q_turn["shoulder_pan"] = q_app["shoulder_pan"]       # 높이 그대로, 팬만 회전
                joint_move(arm, q_turn, "컨베이어 쪽으로 회전")
                if use_app:
                    joint_move(arm, q_app, "컨베이어 위로")
                joint_move(arm, q_drop, "놓는 위치로")
                arm.gripper_open(OPEN_RELEASE_PCT)
                time.sleep(0.3)
                if use_app:
                    joint_move(arm, q_app, "컨베이어 위로 빠지기")
                joint_move(arm, q_turn, "높이 빠지기")          # 블록에서 충분히 떨어진 뒤에만 그리퍼를 닫음
                lap("컨베이어에 놓기", "복귀")
            print("  - 복귀")
            arm.trace_label = "복귀"
            arm.gripper_open(CLOSED_PCT)
            joint_move(arm, home_q, "복귀 자세로")
            lap("복귀")
            rec["phase_s"] = T
            import csv
            tp = log_path.with_name(f"{log_path.stem}_c{cycle}_trace.csv")
            tp.parent.mkdir(parents=True, exist_ok=True)
            with open(tp, "w", newline="") as f:
                csv.writer(f).writerows([["label", "move", "t_move", "phase", "t", "motor", "pos", "cmd", "err", "moving", "load"],
                                         *arm.trace])
            arm.trace = None
            print("  시간: " + ", ".join(f"{k} {v}s" for k, v in T.items()))
            rec["seconds"] = round(time.monotonic() - t0, 1)
            log.append(rec)
            print(f"  완료 {rec['seconds']} s")
    except ArmStop as e:
        print(f"\n[STOP] {e} — 현재 위치에서 정지(유지 중)")
        if arm.last_log:
            import csv
            sp = log_path.with_name(f"{log_path.stem}_stop_motion.csv")
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w", newline="") as f:
                csv.writer(f).writerows([["phase", "t", "motor", "pos", "cmd", "err", "moving", "load"], *arm.last_log])
            print(f"[LOG] {sp}")
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C")
        try:
            arm.hold()
        except Exception:
            pass
    finally:
        loc.stop()
        if log:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), encoding="utf-8")
            print(f"[LOG] {log_path}")
        try:
            ask("\n팔을 받치고 Enter → 토크 OFF > ")
        except (KeyboardInterrupt, EOFError):
            pass
        arm.close(disable_torque=True)


if __name__ == "__main__":
    main()
