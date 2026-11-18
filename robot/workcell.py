"""작업 공간 상수와 카메라→로봇 변환 (robot/workcell.py) — 측정으로 확인된 값만 모아 둔다.

TABLE_Z_M   : 로봇 좌표에서 본 책상 높이. 11/13 토크 ON 호버 시험 4점: -26.4 ~ -28.5 mm (거리 11~21 cm)
BLOCK_M     : 블록 한 변 2.5 cm
GRASP_POINT : 블록을 쥐었을 때 블록 중심의 tool(gripper_frame_link) 좌표.
              URDF 고정 손가락 안쪽 면(x≈+3 mm)에서 블록 반폭만큼 -x, 손끝(z≈+5 mm)에서 7.5 mm 위.
              카메라 보정(s4_solve_camera.py)이 XY를 다시 추정해 calibration/camera_to_base_v2.json에 저장하면 그 값을 쓴다.
"""
import json
from pathlib import Path

import numpy as np

TABLE_Z_M = -0.0275
BLOCK_M = 0.025
GRASP_POINT_DEFAULT = np.array([-0.0095, 0.0, -0.0025])
TRAVEL_ABOVE_TABLE_M = 0.08
WORK_RADIUS_M = (0.10, 0.25)          # pan 축(x=+38.8 mm) 기준 수평 거리
PAN_AXIS_XY = np.array([0.0388, 0.0])
CAMERA_CALIB = Path("calibration/camera_to_base_v2.json")


def block_center_z():
    return TABLE_Z_M + BLOCK_M / 2


def pixel_to_plane(px, H, k1, center, fscale, k2=0.0):
    """픽셀 → (방사 왜곡 보정 k1·r² + k2·r⁴) → 평면 변환 H → 로봇 XY (m). px: (N,2)"""
    px = np.atleast_2d(np.asarray(px, float))
    d = px - center
    r2 = (d ** 2).sum(1, keepdims=True) / fscale ** 2
    pu = center + d * (1 + k1 * r2 + k2 * r2 ** 2)
    hom = np.c_[pu, np.ones(len(pu))] @ np.asarray(H).T
    return hom[:, :2] / hom[:, 2:3]


def load_camera_calib(path=CAMERA_CALIB):
    if not Path(path).is_file():
        return None
    d = json.loads(Path(path).read_text())
    d["grasp_point_in_tool_m"] = np.array(d["grasp_point_in_tool_m"])
    for k in ("R", "t", "H", "center_px"):
        if k in d:
            d[k] = np.array(d[k])
    d.setdefault("mode", "rigid3d")
    return d


def camera_to_robot(calib, detection):
    """카메라 검출 결과 → 로봇 좌표의 블록 '중심' (m). 블록은 책상 위에 놓여 있다고 가정."""
    zc = calib["table_z_m"] + calib["block_m"] / 2
    if calib["mode"] == "homography":
        xy = pixel_to_plane(detection["pixel"], calib["H"], calib["k1"], calib["center_px"], calib["fscale_px"],
                            calib.get("k2", 0.0))[0]
        return np.array([xy[0], xy[1], zc])
    p = calib["R"] @ np.asarray(detection["camera_xyz_m"]) + calib["t"]
    return np.array([p[0], p[1], zc])


def grasp_point():
    c = load_camera_calib()
    return c["grasp_point_in_tool_m"] if c is not None else GRASP_POINT_DEFAULT


def reach_ok(x, y):
    r = float(np.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1]))
    return WORK_RADIUS_M[0] <= r <= WORK_RADIUS_M[1], r
