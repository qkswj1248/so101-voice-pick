"""블록 검출 카메라 (robot/camera.py)

두 종류를 같은 방식으로 쓴다:
  make_camera("top", index=0)   위에서 내려다보는 일반 USB 카메라(OpenCV) — 픽셀 위치만 사용 (권장)
  make_camera("realsense")      RealSense RGB-D — 픽셀 + 깊이로 3D 위치

    with make_camera("top") as cam:
        m, diag = cam.measure("red")        # 여러 프레임 중앙값
        print(m["pixel"], m["camera_xyz_m"])  # top 카메라는 camera_xyz_m = None

픽셀 → 로봇 좌표 변환은 calibration/camera_to_base_v2.json (s4_solve_camera.py가 생성).
"""
from pathlib import Path

import numpy as np

COLOR_RANGES = {   # main.py와 동일 (HSV)
    "red": [((0, 110, 70), (5, 255, 255)), ((175, 110, 70), (179, 255, 255))],
    "orange": [((6, 100, 70), (15, 255, 255))],
    "yellow": [((16, 100, 70), (25, 255, 255))],
    "green": [((36, 70, 50), (85, 255, 255))],
    "blue": [((90, 80, 60), (125, 255, 255))],
    "purple": [((120, 45, 45), (145, 255, 255))],
}
MIN_AREA = 250
MIN_DEPTH_M, MAX_DEPTH_M = 0.10, 1.50
CENTER_RADIUS_PX = 12
EDGE_PX = 3          # 윤곽이 화면 가장자리에 닿으면 블록이 잘린 것 → 중심이 틀어지므로 측정에 쓰지 않음
MAX_AREA = 8000      # 블록 모양 조건: 너무 큰 덩어리, 가늘고 긴 것(케이블), 울퉁불퉁한 것(손) 제외
MAX_ASPECT = 2.0
MIN_SOLIDITY = 0.80
FRAME_OUTLIER_PX = 4.0   # 프레임별 중심이 다수와 이만큼 넘게 다르면 그 프레임은 버림


def color_mask(bgr, color):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def detect_2d(bgr, color, min_area=MIN_AREA):
    """색 영역 윤곽 → 중심 픽셀, 면적, 화면 가장자리 접촉 여부."""
    import cv2
    mask = color_mask(bgr, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > MAX_AREA:
            continue
        (_, (w0, h0), _) = cv2.minAreaRect(c)                 # 회전된 사각형 기준 (대각선 케이블도 걸러짐)
        if max(w0, h0) / max(1.0, min(w0, h0)) > MAX_ASPECT:
            continue
        if area / max(cv2.contourArea(cv2.convexHull(c)), 1.0) < MIN_SOLIDITY:
            continue
        mom = cv2.moments(c)
        if mom["m00"] == 0:
            continue
        u, v = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        bx, by, bw, bh = cv2.boundingRect(c)
        edge = bx <= EDGE_PX or by <= EDGE_PX or bx + bw >= W - EDGE_PX or by + bh >= H - EDGE_PX
        out.append({"pixel": (float(u), float(v)), "camera_xyz_m": None, "area": float(area),
                    "contour": c, "touches_edge": bool(edge), "_mask": mask})
    return out


def detect(bgr, depth_m, intrinsics, color, deproject):
    """RealSense용: detect_2d + 중심 주변 깊이 중앙값으로 3D 위치."""
    import cv2
    out = []
    for d in detect_2d(bgr, color):
        u, v = d["pixel"]
        region = np.zeros_like(d["_mask"])
        cv2.drawContours(region, [d["contour"]], -1, 255, thickness=-1)
        region = cv2.erode(cv2.bitwise_and(region, d["_mask"]), np.ones((3, 3), np.uint8), iterations=2)
        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            continue
        z = depth_m[ys, xs]
        near = ((xs - u) ** 2 + (ys - v) ** 2 <= CENTER_RADIUS_PX ** 2) & np.isfinite(z) & \
               (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
        if np.count_nonzero(near) < 10:
            continue
        d["camera_xyz_m"] = tuple(float(a) for a in deproject(intrinsics, [u, v], float(np.median(z[near]))))
        out.append(d)
    return out


MIN_INLIERS = 25        # RANSAC 후 남은 일치점. 11/17: 60은 조명·소품 변화로 57이 나와 막힘 → 25 + 아래 형태 검사
MIN_MATCHES = 40        # RANSAC 전 원시 매칭 수
MAX_SCALE_DEV = 0.03    # 보정 변환이 크기 ±3% 넘게 바뀌면 오매칭으로 간주
MAX_ROT_DEG = 3.0
MAX_SHIFT_PX = 15       # 고정 카메라: 평소 밀림 ≤2px. 이보다 크면 오매칭이거나 카메라가 실제로 움직임 → 거부


def _gray(bgr):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)      # 밝기·노출 변화 영향 줄이기


class ViewReference:
    """카메라 흔들림 보정용 기준 화면.
    현재 화면과 기준 화면의 배경 특징점(ORB)을 맞춰 '현재 → 기준' 변환(이동+회전+배율)을 구한다.
    로봇 팔·블록·손처럼 움직이는 것은 RANSAC이 걸러냄.
    (11/16 실측: 수집 중 카메라가 5px, 다시 7px 밀림 → 보정 시 검증 오차 7.0 → 4.5 mm)"""

    def __init__(self, bgr):
        import cv2
        self.image = bgr.copy()
        self._orb = cv2.ORB_create(2000)
        self.kp, self.des = self._orb.detectAndCompute(_gray(bgr), None)

    @classmethod
    def load(cls, path):
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return cls(img)

    def save(self, path):
        import cv2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.image)

    def register(self, bgr):
        """(M 2x3 또는 None, inlier 수)"""
        import cv2
        kp, des = self._orb.detectAndCompute(_gray(bgr), None)
        if des is None or self.des is None or len(kp) < MIN_MATCHES:
            return None, 0
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des, self.des)
        if len(m) < MIN_MATCHES:
            return None, len(m)
        src = np.float32([kp[x.queryIdx].pt for x in m])
        dst = np.float32([self.kp[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        n = int(inl.sum()) if inl is not None else 0
        if M is None or n < MIN_INLIERS:
            return None, n
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if abs(scale - 1) > MAX_SCALE_DEV or abs(rot) > MAX_ROT_DEG:   # 고정 카메라에서 나올 수 없는 변환
            return None, n
        if self.shift_px(M) > MAX_SHIFT_PX:
            return None, n
        return M, n

    @staticmethod
    def map(M, pixel):
        return M[:, :2] @ np.asarray(pixel, float) + M[:, 2]

    def shift_px(self, M):
        h, w = self.image.shape[:2]
        pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]])
        return float(np.abs(pts @ M[:, :2].T + M[:, 2] - pts).max())


class _CameraBase:
    kind = "?"

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def frame(self):
        """(bgr, 검출 함수에 넘길 추가 데이터)"""
        raise NotImplementedError

    def find(self, frame, color):
        raise NotImplementedError

    def reference_view(self, n=3):
        """카메라 흔들림 보정용 기준 화면 (배경 특징점)."""
        frames = [self.frame()[0] for _ in range(n)]
        return ViewReference(frames[-1])

    def view_shift(self, ref):
        """기준 화면 대비 현재 화면이 몇 픽셀 밀렸는지 (화면 중앙 기준). 특징점 매칭 실패 시 inf."""
        M, _ = ref.register(self.frame()[0])
        return float("inf") if M is None else ref.shift_px(M)

    def measure(self, color, n_frames=15, expect=1, save_image=None, ref=None, keep_fn=None):
        """n_frames 동안 검출이 정확히 expect개(가장자리 접촉 없음)인 프레임만 모아 중앙값.
        ref(ViewReference)를 주면 카메라가 밀린 만큼을 보정해 '기준 화면 좌표'의 픽셀을 돌려준다."""
        px, xyz, counts, last = [], [], {}, None
        for _ in range(n_frames):
            fr = self.frame()
            dets = self.find(fr, color)
            last = (fr[0], dets)
            if any(d["touches_edge"] for d in dets):          # 가장자리에 걸린 것만 빼고 나머지는 사용
                counts["화면가장자리(제외)"] = counts.get("화면가장자리(제외)", 0) + 1
                dets = [d for d in dets if not d["touches_edge"]]
            if keep_fn is not None:   # 11/17: 작업 영역 밖(팔이 안 닿는 블록 더미 등)은 세지 않음
                dets = [d for d in dets if keep_fn(d["pixel"])]
            counts[len(dets)] = counts.get(len(dets), 0) + 1
            if len(dets) == expect:
                px.append([d["pixel"] for d in dets])
                xyz.append([d["camera_xyz_m"] for d in dets])
        if save_image and last is not None:
            import cv2
            img = last[0].copy()
            for d in last[1]:
                cv2.drawContours(img, [d["contour"]], -1, (255, 255, 255), 2)
                cv2.circle(img, tuple(int(a) for a in d["pixel"]), 4, (255, 255, 255), -1)
            Path(save_image).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_image), img)
        used = len(px)
        diag = {"detections_per_frame": counts, "frames_used": used, "camera": self.kind}
        if used < max(3, n_frames // 3):
            return None, diag
        if expect != 1:
            return [{"pixel": list(p), "camera_xyz_m": x} for p, x in zip(px[-1], xyz[-1])], diag
        p = np.array(px)[:, 0, :]
        keep = np.linalg.norm(p - np.median(p, axis=0), axis=1) <= FRAME_OUTLIER_PX
        diag["frames_outlier"] = int((~keep).sum())
        if keep.sum() < max(3, n_frames // 3):
            diag["error"] = "프레임마다 검출 위치가 크게 달라짐 (다른 빨간 물체/손이 섞임)"
            return None, diag
        p = p[keep]
        xyz = [x for x, k in zip(xyz, keep) if k]
        diag["spread_px"] = float(np.max(np.ptp(p, axis=0)))
        res = {"pixel": np.median(p, axis=0).tolist(), "camera_xyz_m": None}
        if ref is not None:
            M, inl = ref.register(last[0])
            diag["registration_inliers"] = inl
            if M is None:
                # 11/17: 팔이 화면을 가리거나 조명·소품이 바뀌면 매칭이 19~57개로 떨어져 측정 자체가 막힘.
                # 카메라는 하루 종일 밀림 ≤3px였음 → 마지막으로 성공한 보정(없으면 보정 없음)으로 진행하고 경고만
                M = getattr(ref, "last_M", None)
                if M is None:
                    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
                diag["warning"] = "카메라 밀림 보정 실패 → " + ("이전 보정값 사용" if getattr(ref, "last_M", None) is not None
                                                          else "보정 없이 진행")
            else:
                ref.last_M = M
            res["pixel_raw"] = res["pixel"]
            res["pixel"] = ref.map(M, res["pixel"]).tolist()
            diag["drift_px"] = float(np.linalg.norm(np.array(res["pixel"]) - np.array(res["pixel_raw"])))
        if xyz[0][0] is not None:
            x = np.array([f[0] for f in xyz])
            diag["spread_mm"] = float(np.max(np.ptp(x, axis=0)) * 1000)
            res["camera_xyz_m"] = np.median(x, axis=0).tolist()
        return res, diag


class TopCamera(_CameraBase):
    """위에서 내려다보는 일반 USB 카메라 (teleop의 front 카메라와 같은 설정)."""
    kind = "top"

    def __init__(self, index=0, width=640, height=480, fps=30, fourcc="MJPG"):
        self.index, self.size, self.fourcc = index, (width, height, fps), fourcc
        self.cap = None

    def start(self):
        import cv2
        src = int(self.index) if str(self.index).isdigit() else str(self.index)   # 번호 또는 /dev/v4l/by-id/... 경로
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 index {self.index}를 열 수 없음 (다른 프로그램이 쓰는 중이거나 번호가 다름)")
        w, h, fps = self.size
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(30):                       # 자동 노출 안정화
            self.cap.read()
        return self

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frame(self):
        for _ in range(2):                        # 버퍼에 남은 오래된 프레임 버리기
            self.cap.grab()
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("카메라 프레임 읽기 실패")
        return (bgr,)

    def find(self, frame, color):
        return detect_2d(frame[0], color)


class RealSenseCamera(_CameraBase):
    kind = "realsense"

    def __init__(self, width=640, height=480, fps=30):
        self.size = (width, height, fps)
        self.pipeline = None

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        w, h, fps = self.size
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.spatial, self.temporal = rs.spatial_filter(), rs.temporal_filter()
        for _ in range(30):
            self.pipeline.wait_for_frames()
        return self

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def frame(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        depth = self.temporal.process(self.spatial.process(depth)).as_depth_frame()
        bgr = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth.get_units()
        return bgr, depth_m, color.profile.as_video_stream_profile().intrinsics

    def find(self, frame, color):
        bgr, depth_m, intr = frame
        return detect(bgr, depth_m, intr, color, self.rs.rs2_deproject_pixel_to_point)


Camera = RealSenseCamera   # 이전 코드 호환


def make_camera(kind="top", index=0):
    if kind == "top":
        return TopCamera(index=index)
    if kind == "realsense":
        return RealSenseCamera()
    raise ValueError(kind)
