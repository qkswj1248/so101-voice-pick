"""카메라로 블록 찾기 → 로봇 좌표 (robot/perception.py)

    from robot.perception import BlockLocator
    loc = BlockLocator(index="0")          # calibration/camera_to_base_v2.json + camera_reference.png 사용
    with loc:
        b = loc.find("red")
        print(b["xyz_m"], b["yaw_deg"])     # 블록 중심(로봇 좌표, m), 블록 면 방향(도, -45~45)
"""
import math

import numpy as np

from robot import workcell as W
from robot.camera import ViewReference, detect_2d, make_camera


class BlockLocator:
    def __init__(self, index="0", calib_path=W.CAMERA_CALIB):
        self.calib = W.load_camera_calib(calib_path)
        if self.calib is None:
            raise FileNotFoundError(f"{calib_path} 없음 — 4단계 카메라 보정 먼저")
        if self.calib["mode"] != "homography":
            raise ValueError("위 카메라(homography) 보정만 지원")
        self.ref = ViewReference.load(self.calib.get("reference_image") or "calibration/camera_reference.png")
        self.cam = make_camera("top", index)

    def __enter__(self):
        self.cam.start()
        return self

    def __exit__(self, *a):
        self.cam.stop()

    def start(self):
        return self.__enter__()

    def stop(self):
        self.cam.stop()

    def in_workspace(self, px, r_min=0.07, r_max=0.28):
        """화면 픽셀이 팔 주변 작업 영역(팬 축에서 r_min~r_max, 로봇 앞쪽)인지. 같은 색이 영역 밖에 더 있어도 무시."""
        x, y = self._to_robot_xy(np.asarray(px, float))[0]
        r = math.hypot(x - W.PAN_AXIS_XY[0], y - W.PAN_AXIS_XY[1])
        return r_min <= r <= r_max and x > 0.0

    def _to_robot_xy(self, px_ref):
        c = self.calib
        return W.pixel_to_plane(np.atleast_2d(px_ref), c["H"], c["k1"], c["center_px"], c["fscale_px"],
                                c.get("k2", 0.0))

    def find(self, color="red", n_frames=8, save_image=None):
        """하나뿐인 color 블록의 위치·방향. 실패 시 (None, 진단)."""
        m, diag = self.cam.measure(color, n_frames=n_frames, ref=self.ref, save_image=save_image,
                                   keep_fn=self.in_workspace)
        if m is None:
            return None, diag
        xyz = W.camera_to_robot(self.calib, m)
        yaw = self._yaw(color, near_px=np.array(m.get("pixel_raw", m["pixel"])))
        diag["yaw_ok"] = yaw is not None
        if yaw is None:   # 11/17: 예전엔 0°로 대신했는데, 엉뚱한 방향으로 집게 됨 → 실패로 처리
            diag["error"] = "블록 방향 측정 실패 (그 순간 블록이 1개로 안 보임)"
            return None, diag
        return {"xyz_m": xyz, "yaw_deg": yaw, "pixel_ref": m["pixel"],
                "drift_px": diag.get("drift_px", 0.0)}, diag

    def _yaw(self, color, near_px=None, n_try=6, need=3):
        """블록 윤곽의 회전 사각형 방향 → 로봇 좌표계 각도, 큐브라 90° 주기 → [-45, 45).
        11/17: 한 프레임에서 '빨간 것이 정확히 1개'를 요구했더니 팔의 빨간 케이블 등이 가끔 잡혀 실패
        → 여러 프레임에서 위치 측정값(near_px)에 가장 가까운 검출만 쓰고, 90° 주기 평균."""
        import cv2
        vals = []
        for _ in range(n_try):
            bgr = self.cam.frame()[0]
            dets = [d for d in detect_2d(bgr, color) if not d["touches_edge"]]
            if near_px is not None:
                dets = [d for d in dets if np.hypot(*(np.array(d["pixel"]) - near_px)) < 15]
            if len(dets) != 1:
                continue
            M, _ = self.ref.register(bgr)
            if M is None:   # 위치 측정과 같은 대체 규칙 (camera.measure 참고)
                M = getattr(self.ref, "last_M", None)
                if M is None:
                    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            (cx, cy), _, ang = cv2.minAreaRect(dets[0]["contour"])
            a = math.radians(ang)
            p = np.array([[cx, cy], [cx + 15 * math.cos(a), cy + 15 * math.sin(a)]])
            p_ref = np.array([ViewReference.map(M, q) for q in p])
            r = self._to_robot_xy(p_ref)
            vals.append(math.atan2(r[1, 1] - r[0, 1], r[1, 0] - r[0, 0]))
            if len(vals) >= need:
                break
        if not vals:
            return None
        m4 = math.atan2(np.mean(np.sin(4 * np.array(vals))), np.mean(np.cos(4 * np.array(vals)))) / 4
        yaw = math.degrees(m4)
        return (yaw + 45.0) % 90.0 - 45.0
