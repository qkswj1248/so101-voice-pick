"""컨베이어에 블록을 놓을 자세를 손으로 가르쳐 저장 (토크 OFF 상태에서 팔을 직접 옮김).

저장하는 자세 2개:
  1) 접근(approach): 컨베이어 위 5~8cm, 벨트 옆 틀에 안 걸리는 높이
  2) 놓기(drop)    : 그리퍼에 블록을 물었다고 치고, 블록 바닥이 벨트에서 1cm쯤 뜬 위치

python steps/s6_teach_drop.py
→ calibration/drop_conveyor.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.arm import ARM, Arm  # noqa: E402
from robot.kinematics import SO101Kinematics  # noqa: E402
from robot import workcell as W  # noqa: E402

OUT = ROOT / "calibration/drop_conveyor.json"


def teach(arm, kin, name, how):
    print(f"\n[{name}] {how}")
    while True:
        a = input("  Enter = 현재 위치 보기,  s + Enter = 이 자세로 저장 > ").strip().lower()
        q = arm.read_deg()
        p, _ = kin.fk([q[j] for j in ARM])
        print(f"  관절 " + " ".join(f"{j[:5]}={q[j]:+6.1f}" for j in ARM)
              + f"\n  블록 중심 ≈ ({p[0]*100:.1f}, {p[1]*100:.1f}) cm, 책상 위 {(p[2]-W.TABLE_Z_M)*100:.1f} cm")
        if a == "s":
            prob = arm.check_targets_deg({j: q[j] for j in ARM})
            if prob:
                print(f"  [다시] 관절이 측정 범위 끝 5° 안: {prob} — 조금 다른 자세로")
                continue
            return {"q_deg": {j: q[j] for j in ARM}, "block_center_m": p.tolist()}


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
    arm = Arm(port=port).connect()
    kin = SO101Kinematics(contact=W.grasp_point())   # 그리퍼 속 블록 중심
    try:
        input("팔을 손으로 받치고 Enter → 토크 OFF (팔이 힘없이 움직임) > ")
        arm.torque_off()
        approach = teach(arm, kin, "1/2 접근 자세",
                         "그리퍼를 컨베이어 벨트 위 5~8cm로 옮기세요 (벨트 옆 틀보다 높게, 그리퍼는 아래를 향하게).")
        while True:
            drop = teach(arm, kin, "2/2 놓기 자세",
                         "그대로 내려서, 블록을 물었다면 블록 바닥이 벨트에서 1cm쯤 뜰 높이로 옮기세요.")
            dz = approach["block_center_m"][2] - drop["block_center_m"][2]
            if dz >= 0.03:
                break
            print(f"  [다시] 접근 자세가 놓기 자세보다 {dz*100:.1f} cm 높음 — 3cm 이상 내려서 저장해 주세요")
        d = {"approach": approach, "drop": drop, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
             "note": "s6_teach_drop.py로 손으로 가르친 자세 (관절 각도, 로컬 캘리브레이션 기준)"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[저장] {OUT}")
    finally:
        arm.close(disable_torque=True)


if __name__ == "__main__":
    main()
