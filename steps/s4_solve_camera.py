#!/usr/bin/env python3
"""4단계-b: 카메라 → 로봇 변환 계산.

  python steps/s4_solve_camera.py steps/logs/s4_collect_<시각>.jsonl [--save]

로봇 쪽 기준값: 블록을 놓던 순간의 실제 관절값으로 FK한 '그리퍼 속 블록 중심'의 XY
               (블록은 책상 위에 있으므로 높이는 책상 + 블록 높이로 고정)
미지수에 '그리퍼 속 블록 중심 XY'(2개)가 포함됨 — 그리퍼 방향을 바꿔 가며 놓았기 때문에 추정 가능.

모드 (수집 로그에 따라 자동 선택)
  homography (top 카메라)  : 픽셀 (u,v) → 렌즈 왜곡 1항 보정 → 평면 변환 H → 로봇 XY (블록 윗면 평면)
  rigid3d   (realsense)     : R @ 카메라 3D + t = 로봇 3D
검증: 한 점씩 빼고 나머지로 계산한 변환이 빠진 점을 얼마나 맞히는지 (leave-one-out, 수평 오차)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.kinematics import ARM, SO101Kinematics  # noqa: E402
from robot import workcell as W  # noqa: E402

PASS_LOO_MEAN_MM = 4.0
PASS_LOO_MAX_MM = 8.0
MIN_POINTS = 8          # homography 미지수 10~11개 → 점(식 2개씩)이 충분히 많아야 함
MIN_POINTS_K1 = 12
MIN_POINTS_K2 = 14      # 위 카메라는 광각이라 왜곡 2항 필요


def robot_xy(tools, g_xy, gz):
    g = np.array([g_xy[0], g_xy[1], gz])
    return np.array([(T[:3, :3] @ g + T[:3, 3])[:2] for T in tools])


# ---------------------------------------------------------------- homography 모드
def dlt(src, dst):
    A = []
    for (u, v), (x, y) in zip(src, dst):
        A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    h = np.linalg.svd(np.array(A))[2][-1]
    return (h / h[-1]).reshape(3, 3)


def fit_h(px, tools, g0, center, fscale, n_k, fit_g=True):
    """미지수: H 8개 + (그리퍼 속 블록 중심 2개) + 렌즈 왜곡 n_k개(0~2, 점 수에 따라)."""
    gz = g0[2]
    H0 = dlt(px, robot_xy(tools, g0[:2], gz))
    ng = 2 if fit_g else 0

    def unpack(x):
        H = np.append(x[:8], 1.0).reshape(3, 3)
        k = list(x[8 + ng:8 + ng + n_k]) + [0.0] * (2 - n_k)
        return H, k, (x[8:10] if fit_g else g0[:2])

    def resid(x):
        H, k, g = unpack(x)
        return (W.pixel_to_plane(px, H, k[0], center, fscale, k[1]) - robot_xy(tools, g, gz)).ravel() * 1000

    sol = least_squares(resid, np.r_[H0.ravel()[:8], g0[:2] if fit_g else [], np.zeros(n_k)], x_scale="jac")
    H, k, g = unpack(sol.x)
    return {"H": H, "k1": float(k[0]), "k2": float(k[1]), "g": np.r_[g, gz]}, sol.fun.reshape(-1, 2)


def predict_h(model, px, center, fscale):
    return W.pixel_to_plane(np.atleast_2d(px), model["H"], model["k1"], center, fscale)[0]


# ---------------------------------------------------------------- rigid3d 모드
def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fit_3d(cams, tools, g0):
    gz = g0[2]
    top = W.TABLE_Z_M + W.BLOCK_M

    def rob(g_xy):
        xy = robot_xy(tools, g_xy, gz)
        return np.c_[xy, np.full(len(xy), top)]

    R0, t0 = kabsch(cams, rob(g0[:2]))

    def resid(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        return ((cams @ R.T + x[3:6]) - rob(x[6:8])).ravel() * 1000

    sol = least_squares(resid, np.r_[Rotation.from_matrix(R0).as_rotvec(), t0, g0[:2]])
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    return {"R": R, "t": sol.x[3:6], "g": np.r_[sol.x[6:8], gz]}, sol.fun.reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--save", action="store_true", help="합격 시 calibration/camera_to_base_v2.json 저장")
    ap.add_argument("--exclude", default="", help='빼고 풀 점 번호, 예: "3,7" (화면 가장자리 등)')
    args = ap.parse_args()
    excl = {int(v) for v in args.exclude.replace(",", " ").split()}

    recs = [json.loads(l) for l in Path(args.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("camera") and r["index"] not in excl]
    if excl:
        print(f"[제외] 점 {sorted(excl)}")
    pointer = recs and recs[0].get("mode") == "pointer"
    if len(recs) < MIN_POINTS:
        sys.exit(f"카메라 측정이 있는 점이 {len(recs)}개 — 최소 {MIN_POINTS}개 필요")
    kin = SO101Kinematics()
    g0 = np.array(recs[0].get("grasp_point_default", W.GRASP_POINT_DEFAULT))
    if pointer:
        # 손끝 기준 수집: 블록 중심 = 손끝 바로 아래 → 그리퍼 속 위치 추정 불필요 (tool 변환 대신 손끝 XY 직접 사용)
        tools = [np.r_[np.c_[np.eye(3), [r["tip_xy_m"][0], r["tip_xy_m"][1], 0.0]], [[0, 0, 0, 1]]] for r in recs]
        g0 = np.array([0.0, 0.0, 0.0])
    else:
        tools = [kin.tool_pose([r["q_release_deg"][j] for j in ARM]) for r in recs]
    mode = "rigid3d" if recs[0]["camera"].get("camera_xyz_m") else "homography"
    center, fscale = np.array([320.0, 240.0]), 640.0
    print(f"모드: {mode}{' (손끝 기준)' if pointer else ''}, 점 {len(recs)}개")

    if mode == "homography":
        px = np.array([r["camera"]["pixel"] for r in recs])
        # 손끝 기준은 미지수가 2개 적음. 왜곡 항은 1개까지만 (2항은 15점에서 불안정했음 — 11/16 검증)
        n_k = 1 if len(recs) >= MIN_POINTS_K1 else 0
        print(f"렌즈 왜곡 보정 항: {n_k}개 (점 {MIN_POINTS_K1}개 이상 1항, {MIN_POINTS_K2}개 이상 2항)")
        fit = lambda idx: fit_h(px[idx], [tools[k] for k in idx], g0, center, fscale, n_k, not pointer)  # noqa: E731
        pred = lambda m, i: predict_h(m, px[i], center, fscale)                          # noqa: E731
    else:
        cams = np.array([r["camera"]["camera_xyz_m"] for r in recs])
        fit = lambda idx: fit_3d(cams[idx], [tools[k] for k in idx], g0)               # noqa: E731
        pred = lambda m, i: (m["R"] @ cams[i] + m["t"])[:2]                             # noqa: E731

    allidx = list(range(len(recs)))
    model, res = fit(allidx)
    err = np.linalg.norm(res[:, :2], axis=1)
    if not pointer:
        print(f"그리퍼 속 블록 중심 (tool 좌표) = {np.round(model['g']*1000, 1)} mm  (초기값 {np.round(g0*1000, 1)})")
    if mode == "homography":
        print(f"렌즈 왜곡 계수 k1 = {model['k1']:+.3f}, k2 = {model['k2']:+.3f}")
    print(f"{'#':>3} {'목표(cm)':>12} {'roll':>5} | {'dX':>6} {'dY':>6} | {'|e|':>5}  (mm)")
    for r, e in zip(recs, res):
        print(f"{r['index']:>3} ({r['target_xy_m'][0]*100:4.1f},{r['target_xy_m'][1]*100:5.1f}) "
              f"{r['roll_off_deg']:+5.0f} | {e[0]:6.1f} {e[1]:6.1f} | {np.linalg.norm(e[:2]):5.1f}")
    print(f"적합 오차(수평): 평균 {err.mean():.1f} mm, 최대 {err.max():.1f} mm")

    loo = []
    for i in allidx:
        keep = [k for k in allidx if k != i]
        m, _ = fit(keep)
        true = robot_xy([tools[i]], m["g"][:2], m["g"][2])[0]
        loo.append(np.linalg.norm(pred(m, i) - true) * 1000)
    loo = np.array(loo)
    print("\n=== 검증 (한 점씩 빼고 예측, 수평 오차) ===")
    print("  " + "  ".join(f"{r['index']}:{v:.1f}" for r, v in zip(recs, loo)))
    print(f"  평균 {loo.mean():.1f} mm, 최대 {loo.max():.1f} mm  "
          f"(합격 기준: 평균 ≤ {PASS_LOO_MEAN_MM}, 최대 ≤ {PASS_LOO_MAX_MM})")
    passed = loo.mean() <= PASS_LOO_MEAN_MM and loo.max() <= PASS_LOO_MAX_MM
    print("  → 합격" if passed else "  → 불합격: 오차가 큰 점의 사진(steps/logs/*_p<번호>.png)과 검출 상태 확인")

    if args.save:
        if not passed:
            print("[저장 안 함] 불합격 결과는 저장하지 않습니다.")
            return
        gp = W.GRASP_POINT_DEFAULT if pointer else model["g"]
        ref_png = Path(args.log).with_name(Path(args.log).stem + "_reference.png")
        if ref_png.is_file():
            import shutil
            shutil.copy(ref_png, ROOT / "calibration/camera_reference.png")
            print("[저장] calibration/camera_reference.png (카메라 흔들림 보정 기준 화면)")
        out = {"mode": mode, "grasp_point_in_tool_m": np.asarray(gp).tolist(),
               "collection": "pointer" if pointer else "place",
               "reference_image": "calibration/camera_reference.png" if ref_png.is_file() else None,
               "table_z_m": W.TABLE_Z_M, "block_m": W.BLOCK_M,
               "fit_mean_mm": float(err.mean()), "loo_mean_mm": float(loo.mean()), "loo_max_mm": float(loo.max()),
               "n_points": len(recs), "source_log": str(args.log), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "camera_kind": recs[0].get("camera_kind", "realsense" if mode == "rigid3d" else "top")}
        if mode == "homography":
            out.update({"H": model["H"].tolist(), "k1": model["k1"], "k2": model["k2"], "center_px": center.tolist(),
                        "fscale_px": fscale,
                        "note": "픽셀 → 왜곡 보정 → H → 로봇 XY(블록 윗면 평면). 카메라/로봇/클램프를 움직이면 다시 수집."})
        else:
            out.update({"R": model["R"].tolist(), "t": model["t"].tolist(),
                        "note": "R @ 카메라3D(블록 윗면 중심) + t = 로봇 좌표. 카메라/로봇/클램프를 움직이면 다시 수집."})
        (ROOT / W.CAMERA_CALIB).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / W.CAMERA_CALIB).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[저장] {W.CAMERA_CALIB}")


if __name__ == "__main__":
    main()
