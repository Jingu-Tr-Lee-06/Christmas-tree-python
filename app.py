import base64
import json
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Christmas Tree 2026 | JH's SW Lab",
    page_icon="🎄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================================
# 1. 상위폴더/Music/song.mp3 탐색 및 Base64 캐싱 로드
# =====================================================================
def find_music_file() -> Path | None:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir.parent / "Music" / "song.mp3",   # 1순위: 상위폴더/Music/song.mp3
        base_dir / "Music" / "song.mp3",          # 2순위: 현재폴더/Music/song.mp3
        base_dir.parent / "music" / "song.mp3",   # 대소문자 구분 환경 대비
        base_dir / "music" / "song.mp3",
        Path.cwd().parent / "Music" / "song.mp3",
        Path.cwd() / "Music" / "song.mp3",
    ]
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_music_base64(path_str: str, mtime_ns: int) -> str:
    data = Path(path_str).read_bytes()
    return base64.b64encode(data).decode("ascii")


# =====================================================================
# 2. 3D 크리스마스 트리 점군(Point Cloud) 생성 엔진 (NumPy + 캐싱)
# =====================================================================
NEEDLE, TRUNK, BALL, LIGHT, GARLAND, GIFT, RIBBON, SNOWCAP, GROUND, STAR = range(10)
YB, YT = 0.02, 2.35


def envelope(y):
    """높이 y에 따른 트리 원뿔 외곽 반지름"""
    t = np.clip((y - YB) / (YT - YB), 0.0, 1.0)
    return 0.88 * (1.0 - t) ** 1.08


def fib_sphere(n, r):
    """피보나치 격자를 이용해 구 표면에 균일하게 분포된 3D 점 생성"""
    i = np.arange(n, dtype=np.float32)
    u = (i + 0.5) / n
    th = np.arccos(1.0 - 2.0 * u)
    ph = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [
            r * np.sin(th) * np.cos(ph),
            r * np.cos(th),
            r * np.sin(th) * np.sin(ph),
        ],
        axis=-1,
    )


class Parts:
    """3D 점 좌표(P), 색상(C), 법선(N), 종류(K), 위상(PH), 생성 시점(BD) 컨테이너"""

    def __init__(self):
        self.P, self.C, self.N, self.K, self.PH, self.BD = [], [], [], [], [], []

    def add(self, p, c, kind, nrm=None, ph=0.0, bd=0.0):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 3)
        m = len(p)
        if m == 0:
            return
        c = np.broadcast_to(np.asarray(c, dtype=np.float32), (m, 3)).copy()
        if nrm is None:
            nrm = p.copy()
            nrm[:, 1] = 0.35
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
        else:
            nrm = np.broadcast_to(np.asarray(nrm, dtype=np.float32), (m, 3)).copy()

        self.P.append(p)
        self.C.append(c)
        self.N.append(nrm)
        self.K.append(np.full(m, kind, dtype=np.uint8))
        self.PH.append(np.broadcast_to(np.asarray(ph, dtype=np.float32), (m,)).copy())
        self.BD.append(np.broadcast_to(np.asarray(bd, dtype=np.float32), (m,)).copy())

    def done(self):
        return [
            np.concatenate(x, axis=0)
            for x in (self.P, self.C, self.N, self.K, self.PH, self.BD)
        ]


def add_gift_box(q, rng, center_xz, size_xyz, yaw, box_col, rib_col, bd_base):
    """선물 상자 5면과 십자 리본, 상단 리본 매듭 생성"""
    gx, gz = center_xz
    sx, sy, sz = size_xyz
    rib_w = min(sx, sz) * 0.16
    faces = []

    u, w = np.meshgrid(np.linspace(-sx / 2, sx / 2, 13), np.linspace(-sz / 2, sz / 2, 13))
    faces.append(
        (
            u.ravel(),
            np.full(u.size, sy),
            w.ravel(),
            [0.0, 1.0, 0.0],
            (np.abs(u.ravel()) < rib_w) | (np.abs(w.ravel()) < rib_w),
        )
    )

    u, v = np.meshgrid(np.linspace(-sx / 2, sx / 2, 13), np.linspace(0.0, sy, 10))
    for sgn in (-1.0, 1.0):
        faces.append(
            (u.ravel(), v.ravel(), np.full(u.size, sgn * sz / 2), [0.0, 0.0, sgn], np.abs(u.ravel()) < rib_w)
        )

    w, v = np.meshgrid(np.linspace(-sz / 2, sz / 2, 13), np.linspace(0.0, sy, 10))
    for sgn in (-1.0, 1.0):
        faces.append(
            (np.full(w.size, sgn * sx / 2), v.ravel(), w.ravel(), [sgn, 0.0, 0.0], np.abs(w.ravel()) < rib_w)
        )

    cy, sy_rot = np.cos(yaw), np.sin(yaw)
    for lx, ly, lz, fnrm, is_rib in faces:
        rx = lx * cy + lz * sy_rot + gx
        rz = -lx * sy_rot + lz * cy + gz
        pts = np.stack([rx, ly, rz], axis=-1)
        rnrm = [fnrm[0] * cy + fnrm[2] * sy_rot, fnrm[1], -fnrm[0] * sy_rot + fnrm[2] * cy]
        bd = bd_base + 0.18 * (ly / max(sy, 0.01))
        q.add(pts[~is_rib], box_col, GIFT, nrm=rnrm, bd=bd[~is_rib])
        q.add(pts[is_rib], rib_col, RIBBON, nrm=rnrm, bd=bd[is_rib])

    t_bow = np.linspace(0.0, 2.0 * np.pi, 70)
    r_bow = rib_w * 2.1 * np.sin(2.0 * t_bow)
    bx = r_bow * np.cos(t_bow)
    bz = r_bow * np.sin(t_bow) * 0.55
    by = sy + np.abs(r_bow) * 0.55 + 0.008
    bow_pts = np.stack([bx * cy + bz * sy_rot + gx, by, -bx * sy_rot + bz * cy + gz], axis=-1)
    bow_pts += rng.normal(0.0, 0.003, bow_pts.shape)
    q.add(bow_pts, rib_col, RIBBON, nrm=[0.0, 1.0, 0.0], bd=bd_base + 0.22)


@st.cache_data(show_spinner=False)
def build_scene_json() -> str:
    rng = np.random.default_rng(7)
    q = Parts()
    tips, sub_tips = [], []

    # 1. 눈 덮인 바닥
    ng = 5500
    rg = np.sqrt(rng.uniform(0.0, 1.0, ng)) * 1.35
    ag = rng.uniform(0.0, 2.0 * np.pi, ng)
    yg = -0.02 + 0.028 * np.cos(rg * 3.5) * np.exp(-rg * 0.7) + rng.normal(0.0, 0.004, ng)
    pg = np.stack([rg * np.cos(ag), yg, rg * np.sin(ag)], axis=-1)
    shade_g = np.clip(0.42 + 0.58 * (rg / 0.85), 0.42, 1.0) * np.clip((1.38 - rg) / 0.26, 0.25, 1.0)
    q.add(pg, np.outer(shade_g, [0.78, 0.86, 0.98]), GROUND, nrm=[0.0, 1.0, 0.0], bd=0.75 * (rg / 1.35))

    # 2. 나무 기둥
    nt = 2200
    yt = rng.uniform(0.0, YT * 0.92, nt)
    rt = 0.095 * (1.0 - 0.78 * (yt / YT))
    at = rng.uniform(0.0, 2.0 * np.pi, nt)
    pt = np.stack([rt * np.cos(at), yt, rt * np.sin(at)], axis=-1)
    nt_nrm = np.stack([np.cos(at), np.zeros_like(at), np.sin(at)], axis=-1)
    q.add(
        pt,
        np.array([0.38, 0.22, 0.11]) * rng.uniform(0.82, 1.18, (nt, 1)),
        TRUNK,
        nrm=nt_nrm,
        bd=0.30 + 2.50 * (yt / YT),
    )

    # 3. 솔잎 가지 26층 & 가지 끝 눈
    tiers = np.linspace(YB + 0.02, YT - 0.04, 26)
    for i, y in enumerate(tiers):
        nb = int(np.interp(i, [0, 25], [15, 5]))
        R = envelope(y)
        base_a = i * 2.399963
        tier_t = (y - YB) / (YT - YB)

        for b in range(nb):
            a = base_a + 2.0 * np.pi * b / nb + rng.normal(0.0, 0.08)
            L = R * rng.uniform(0.86, 1.08)
            n = int(250 * L) + 48
            t = rng.uniform(0.0, 1.0, n) ** 0.65
            d = t * L

            ca, sa = np.cos(a), np.sin(a)
            dir_v = np.array([ca, 0.0, sa])
            side = np.array([-sa, 0.0, ca])

            spread = 0.014 + 0.115 * L * (t**0.75)
            lat = rng.normal(0.0, 1.0, n) * spread
            vert = rng.normal(0.0, 0.55, n) * spread
            droop = 0.16 * (d**1.35)

            p_br = (
                d[:, None] * dir_v
                + lat[:, None] * side
                + np.stack([np.zeros(n), y - droop + vert, np.zeros(n)], axis=-1)
            )
            g_base = np.array([0.05, 0.22, 0.07])
            g_tip = np.array([0.16, 0.58, 0.21])
            col = np.clip(
                g_base + (g_tip - g_base) * (t[:, None] ** 0.85) + rng.normal(0.0, 0.025, (n, 3)),
                0.02,
                1.0,
            )
            bd_br = 0.35 + 2.55 * tier_t + 0.25 * t

            nrm_br = 0.60 * dir_v + np.array([0.0, 0.68, 0.0]) + 0.35 * np.sign(lat)[:, None] * side
            nrm_br /= np.linalg.norm(nrm_br, axis=1, keepdims=True) + 1e-6
            q.add(p_br, col, NEEDLE, nrm=nrm_br, bd=bd_br)

            if i < 24:
                ns = int(25 * L) + 4
                ts = rng.uniform(0.70, 1.0, ns)
                ds = ts * L
                lats = rng.normal(0.0, 0.6, ns) * (0.014 + 0.09 * L * ts)
                p_sn = (
                    ds[:, None] * dir_v
                    + lats[:, None] * side
                    + np.stack([np.zeros(ns), y - 0.16 * (ds**1.35) + 0.017, np.zeros(ns)], axis=-1)
                )
                q.add(p_sn, [0.86, 0.93, 1.0], SNOWCAP, nrm=[0.0, 1.0, 0.0], bd=float(bd_br.max()))

            tips.append(L * dir_v + np.array([0.0, y - 0.16 * (L**1.35), 0.0]))
            mid_L = L * rng.uniform(0.60, 0.88)
            sub_tips.append(
                mid_L * dir_v
                + rng.normal(0.0, 0.02, 3)
                + np.array([0.0, y - 0.16 * (mid_L**1.35), 0.0])
            )

    tips_arr = np.array(tips, dtype=np.float32)
    all_anchors = np.vstack([tips_arr, np.array(sub_tips, dtype=np.float32)])

    # 4. 나선형 가랜드 줄
    for strand, col_g in enumerate(([1.0, 0.82, 0.30], [0.96, 0.68, 0.52])):
        ug = np.linspace(0.0, 1.0, 920)
        yg = 0.16 + ug * (YT - 0.38)
        rg = envelope(yg) * 0.94
        th_g = ug * 2.0 * np.pi * 5.2 + strand * np.pi
        sag = 0.048 * (1.0 - np.abs(np.sin(th_g * 4.5))) * (1.0 - 0.55 * ug)
        pg = np.stack([rg * np.cos(th_g), yg - sag, rg * np.sin(th_g)], axis=-1) + rng.normal(
            0.0, 0.006, (920, 3)
        )
        q.add(pg, col_g, GARLAND, bd=2.85 + 0.85 * ug)

    # 5. 오너먼트 볼
    ball_palette = np.array(
        [
            [0.96, 0.16, 0.20],
            [1.00, 0.78, 0.16],
            [0.20, 0.55, 0.98],
            [0.72, 0.92, 1.00],
            [0.94, 0.26, 0.68],
        ]
    )
    valid_tips = tips_arr[(tips_arr[:, 1] > 0.16) & (tips_arr[:, 1] < YT - 0.20)]
    chosen_balls = []
    sph_unit = fib_sphere(52, 1.0)
    for idx in rng.permutation(len(valid_tips)):
        cand = valid_tips[idx]
        if all(np.linalg.norm(cand - prev) > 0.20 for prev in chosen_balls):
            chosen_balls.append(cand)
            r_ball = rng.uniform(0.046, 0.058)
            center = cand - np.array([0.0, r_ball * 0.85, 0.0])
            b_col = ball_palette[len(chosen_balls) % len(ball_palette)]
            q.add(
                center + sph_unit * r_ball,
                b_col,
                BALL,
                nrm=sph_unit,
                bd=3.0 + 0.85 * ((cand[1] - YB) / (YT - YB)),
            )
            if len(chosen_balls) >= 54:
                break

    # 6. 꼬마전구 175개
    n_lights = 175
    light_pts = all_anchors[
        rng.choice(len(all_anchors), size=min(n_lights, len(all_anchors)), replace=False)
    ]
    light_palette = np.array(
        [
            [1.00, 0.90, 0.48],
            [1.00, 0.38, 0.30],
            [0.55, 0.95, 1.00],
            [1.00, 0.97, 0.85],
            [0.45, 1.00, 0.58],
        ]
    )
    q.add(
        light_pts,
        light_palette[rng.integers(0, len(light_palette), size=len(light_pts))],
        LIGHT,
        ph=rng.uniform(0.0, 2.0 * np.pi, size=len(light_pts)),
        bd=3.10 + 0.85 * np.clip((light_pts[:, 1] - YB) / (YT - YB), 0.0, 1.0),
    )

    # 7. 선물 상자 6개
    gift_specs = [
        ((0.68, 0.22), (0.26, 0.20, 0.24), 0.35, [0.88, 0.16, 0.18], [1.00, 0.82, 0.24]),
        ((-0.58, 0.48), (0.24, 0.25, 0.22), 1.10, [0.16, 0.38, 0.82], [0.92, 0.95, 1.00]),
        ((0.15, -0.78), (0.30, 0.17, 0.25), -0.45, [0.14, 0.58, 0.28], [0.95, 0.22, 0.24]),
        ((-0.76, -0.32), (0.22, 0.19, 0.22), 0.80, [0.92, 0.72, 0.18], [0.85, 0.14, 0.22]),
        ((0.52, -0.58), (0.21, 0.22, 0.20), -1.05, [0.68, 0.20, 0.58], [1.00, 0.90, 0.72]),
        ((-0.12, 0.82), (0.25, 0.16, 0.21), 0.20, [0.15, 0.62, 0.68], [1.00, 0.84, 0.28]),
    ]
    for g_i, (xz, sz, yaw, bcol, rcol) in enumerate(gift_specs):
        add_gift_box(q, rng, xz, sz, yaw, bcol, rcol, bd_base=3.55 + g_i * 0.10)

    # 8. 꼭대기 별 & 빛줄기
    star_y = YT + 0.11
    ns_pts = 950
    ang_s = rng.uniform(0.0, 2.0 * np.pi, ns_pts)
    sector = np.abs(((ang_s - np.pi / 2.0) % (2.0 * np.pi / 5.0)) - np.pi / 5.0)
    r_bound = 0.052 + (0.145 - 0.052) * ((sector / (np.pi / 5.0)) ** 1.4)
    rad_s = np.sqrt(rng.uniform(0.0, 1.0, ns_pts)) * r_bound
    zs = rng.uniform(-1.0, 1.0, ns_pts) * 0.042 * (1.0 - rad_s / 0.15)
    ps = np.stack([rad_s * np.cos(ang_s), star_y + rad_s * np.sin(ang_s), zs], axis=-1)
    cs = np.clip(
        np.array([1.0, 0.96, 0.72]) + (1.0 - rad_s[:, None] / 0.15) * np.array([0.0, 0.04, 0.25]),
        0.0,
        1.0,
    )
    q.add(ps, cs, STAR, ph=rng.uniform(0.0, 2.0 * np.pi, ns_pts), bd=4.15)

    for ray_dir, ray_len in [
        ([1, 0, 0], 0.26),
        ([-1, 0, 0], 0.26),
        ([0, 1, 0], 0.28),
        ([0, -1, 0], 0.22),
        ([0, 0, 1], 0.24),
        ([0, 0, -1], 0.24),
        ([0.7, 0.7, 0], 0.19),
        ([-0.7, 0.7, 0], 0.19),
        ([0.7, -0.7, 0], 0.17),
        ([-0.7, -0.7, 0], 0.17),
    ]:
        rd = np.asarray(ray_dir, dtype=np.float32)
        rd /= np.linalg.norm(rd)
        u_ray = np.linspace(0.05, ray_len, 28)
        pray = np.array([0.0, star_y, 0.0]) + u_ray[:, None] * rd
        q.add(
            pray,
            np.outer((1.0 - u_ray / ray_len) ** 0.7, [1.0, 0.95, 0.65]),
            STAR,
            ph=u_ray * 12.0,
            bd=4.25,
        )

    P, C, N, K, PH, BD = q.done()
    b64 = lambda arr: base64.b64encode(arr.tobytes()).decode("ascii")

    return json.dumps(
        {
            "n": int(len(P)),
            "lights": int(len(light_pts)),
            "P": b64(np.clip(np.round(P * 4000.0), -32767, 32767).astype("<i2")),
            "C": b64(np.clip(np.round(C * 255.0), 0, 255).astype(np.uint8)),
            "N": b64(np.clip(np.round(N * 127.0), -127, 127).astype(np.int8)),
            "K": b64(K.astype(np.uint8)),
            "PH": b64(
                np.clip(np.round((PH % (2 * np.pi)) / (2 * np.pi) * 255.0), 0, 255).astype(np.uint8)
            ),
            "BD": b64(np.clip(np.round(BD * 1000.0), 0, 65535).astype("<u2")),
        }
    )


# =====================================================================
# 3. 풀스크린 반응형 웹 애플리케이션 HTML/CSS/JS 조립
# =====================================================================
def build_responsive_app_html(scene_json: str, audio_b64: str) -> str:
    html = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Christmas Tree 2026 | JH's SW Lab</title>
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    -webkit-tap-highlight-color: transparent;
  }
  html, body {
    width: 100vw;
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    background: #070a10;
    color: #e6edf3;
    font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    user-select: none;
  }

  /* =========================================================
     0) 첫 접속 '입장하시겠습니까?' 모달 (지연 없는 즉시 음악 재생용)
     ========================================================= */
  #entry-gate {
    position: fixed;
    inset: 0;
    z-index: 3000;
    background: radial-gradient(circle at 50% 45%, rgba(16, 24, 39, 0.88) 0%, rgba(5, 8, 14, 0.96) 100%);
    backdrop-filter: blur(16px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    transition: opacity 0.32s ease, visibility 0.32s ease;
  }
  #entry-gate.hidden {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
  }
  .gate-modal {
    width: min(410px, 100%);
    background: rgba(15, 22, 34, 0.95);
    border: 1px solid rgba(71, 85, 105, 0.75);
    border-radius: 22px;
    padding: 34px 28px 28px;
    text-align: center;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.65);
  }
  .gate-icon {
    font-size: 46px;
    line-height: 1;
    margin-bottom: 14px;
    filter: drop-shadow(0 0 14px rgba(251, 191, 36, 0.35));
  }
  .gate-title {
    font-size: 1.42rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
  }
  .gate-desc {
    font-size: 0.93rem;
    color: #94a3b8;
    line-height: 1.55;
    margin-bottom: 24px;
  }
  .gate-btn {
    width: 100%;
    border: none;
    border-radius: 13px;
    padding: 15px 20px;
    font-size: 1.05rem;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 8px 24px rgba(22, 163, 74, 0.4);
    cursor: pointer;
    transition: transform 0.14s ease, box-shadow 0.18s ease, filter 0.18s ease;
    font-family: inherit;
  }
  .gate-btn:hover {
    filter: brightness(1.08);
    box-shadow: 0 10px 28px rgba(34, 197, 94, 0.55);
  }
  .gate-btn:active {
    transform: scale(0.97);
  }
  .gate-warn {
    margin-top: 12px;
    font-size: 0.78rem;
    color: #fca5a5;
    display: none;
  }

  /* 전체 메인 래퍼 */
  #main-view {
    position: relative;
    width: 100vw;
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
  }

  /* =========================================================
     1) 데스크톱 레이아웃 (769px 이상)
        - 좌측 끝부터 2/3 (66.6667vw): 트리 영역
        - 우측 1/3 여백의 정중앙: 솔크 질문창 배치
     ========================================================= */
  #tree-stage {
    position: fixed;
    top: 0;
    left: 0;
    width: 66.6667vw;
    height: calc(100vh - 46px);
    height: calc(100dvh - 46px);
    background: radial-gradient(circle at 50% 56%, #0f1624 0%, #070a10 78%);
    border-right: 1px solid rgba(36, 48, 68, 0.65);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    z-index: 1;
  }

  #cv-wrap {
    flex: 1;
    width: 100%;
    height: 100%;
    position: relative;
    overflow: hidden;
  }

  canvas#cv {
    width: 100%;
    height: 100%;
    display: block;
    image-rendering: pixelated;
  }

  .hud-bar {
    height: 32px;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 12px;
    color: #8796af;
    background: rgba(9, 13, 20, 0.88);
    border-top: 1px solid #161f2e;
    flex-shrink: 0;
  }

  /* 우측 1/3 여백의 정중앙(가로/세로 중앙)에 질문창 정렬 */
  #ui-zone {
    position: fixed;
    top: 0;
    left: 66.6667vw;
    width: 33.3333vw;
    height: calc(100vh - 46px);
    height: calc(100dvh - 46px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    z-index: 10;
  }

  .prompt-card {
    width: min(360px, 100%);
    background: rgba(15, 20, 30, 0.92);
    border: 1px solid rgba(51, 65, 85, 0.78);
    border-radius: 18px;
    padding: 28px 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(14px);
    transition: all 0.25s ease;
  }

  .card-badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.35);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 12px;
  }

  .card-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
  }

  .card-desc {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 22px;
  }

  .btn-group {
    display: flex;
    flex-direction: column;
    gap: 11px;
  }

  .choice-btn {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.14s ease, box-shadow 0.18s ease, background 0.18s ease;
    font-family: inherit;
  }

  .choice-btn:active {
    transform: scale(0.97);
  }

  .btn-solo {
    background: linear-gradient(135deg, #e11d48 0%, #be123c 100%);
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(225, 29, 72, 0.35);
  }
  .btn-solo:hover {
    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
    box-shadow: 0 8px 25px rgba(244, 63, 94, 0.5);
  }

  .btn-couple {
    background: rgba(30, 41, 59, 0.85);
    color: #cbd5e1;
    border: 1px solid rgba(71, 85, 105, 0.7);
  }
  .btn-couple:hover {
    background: rgba(51, 65, 85, 0.9);
    color: #ffffff;
  }

  .result-banner {
    display: none;
    margin-top: 16px;
    padding: 14px 16px;
    border-radius: 12px;
    background: rgba(16, 185, 129, 0.16);
    border: 1px solid rgba(52, 211, 153, 0.55);
    color: #6ee7b7;
    font-size: 1.06rem;
    font-weight: 800;
    text-align: center;
    animation: popIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .result-banner.show {
    display: block;
  }

  @keyframes popIn {
    0% { opacity: 0; transform: translateY(6px) scale(0.96); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
  }

  /* 상단 토스트 알림 */
  #solo-toast {
    position: fixed;
    top: 22px;
    left: 50%;
    transform: translateX(-50%) translateY(-20px);
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(52, 211, 153, 0.7);
    color: #6ee7b7;
    padding: 12px 24px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 1rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.28s ease, transform 0.28s ease;
    z-index: 1200;
    white-space: nowrap;
  }
  #solo-toast.visible {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  /* 파티클(컨페티/별빛) 전용 풀스크린 오버레이 캔버스 */
  canvas#fx-canvas {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    height: 100dvh;
    pointer-events: none;
    z-index: 1100;
  }

  /* 최하단 JH's SW Lab + 음원 출처 고정 푸터 */
  .jh-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100vw;
    height: 46px;
    background: rgba(8, 11, 17, 0.96);
    border-top: 1px solid #1e293b;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 0 16px;
    font-size: 0.84rem;
    z-index: 2500;
  }
  .footer-brand {
    font-weight: 800;
    color: #cbd5e1;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }
  .footer-sep {
    color: #475569;
  }
  .footer-credit {
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .footer-credit a {
    color: #93c5fd;
    text-decoration: underline;
    text-underline-offset: 3px;
    font-weight: 600;
    transition: color 0.15s ease;
  }
  .footer-credit a:hover {
    color: #dbeafe;
  }

  /* =========================================================
     2) '솔크가 아닙니다' 클릭 시 전환되는 404 페이지
        (음악이 끊기지 않도록 페이지 리로드 없이 전환)
     ========================================================= */
  #page-404 {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: calc(100vh - 46px);
    height: calc(100dvh - 46px);
    background: #ffffff;
    color: #111111;
    z-index: 2200;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding-top: 15vh;
    text-align: center;
    font-family: Georgia, "Times New Roman", serif;
    user-select: text;
  }
  #page-404.active {
    display: flex;
  }
  #page-404 h1 {
    font-size: clamp(32px, 5vw, 46px);
    font-weight: bold;
    color: #000000;
    margin-bottom: 16px;
  }
  #page-404 hr {
    border: none;
    border-top: 1px solid #cccccc;
    width: min(86%, 920px);
    margin: 14px auto 18px;
  }
  #page-404 .sub-404 {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 14px;
    color: #444444;
    margin-bottom: 22px;
  }
  #page-404 .couple-msg {
    display: inline-block;
    font-family: "Pretendard", -apple-system, sans-serif;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #b91c1c;
    font-weight: 800;
    font-size: 1.05rem;
    padding: 12px 24px;
    border-radius: 10px;
  }

  /* =========================================================
     3) 휴대폰 / 모바일 화면 크기 (768px 이하)
        - 트리: 풀화면
        - 하단: 솔크 여부 선택창 배치
     ========================================================= */
  @media (max-width: 768px) {
    #tree-stage {
      width: 100vw;
      height: calc(100vh - 54px);
      height: calc(100dvh - 54px);
      border-right: none;
    }

    .hud-bar {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 28px;
      font-size: 10.5px;
      padding: 0 12px;
      background: linear-gradient(180deg, rgba(7, 10, 16, 0.84) 0%, rgba(7, 10, 16, 0) 100%);
      border-top: none;
      z-index: 5;
      pointer-events: none;
    }

    #ui-zone {
      top: auto;
      left: 0;
      bottom: 62px;
      width: 100vw;
      height: auto;
      justify-content: center;
      padding: 0 12px;
    }

    .prompt-card {
      width: 100%;
      max-width: 480px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(12, 17, 26, 0.82);
      border: 1px solid rgba(71, 85, 105, 0.65);
      backdrop-filter: blur(12px);
    }

    .card-badge {
      display: none;
    }

    .card-title {
      font-size: 1rem;
      text-align: center;
      margin-bottom: 3px;
    }

    .card-desc {
      font-size: 0.78rem;
      text-align: center;
      margin-bottom: 11px;
    }

    .btn-group {
      flex-direction: row;
      gap: 9px;
    }

    .choice-btn {
      padding: 11px 10px;
      font-size: 0.9rem;
      border-radius: 10px;
    }

    .result-banner {
      margin-top: 10px;
      padding: 9px 12px;
      font-size: 0.92rem;
    }

    #page-404 {
      height: calc(100vh - 54px);
      height: calc(100dvh - 54px);
    }

    .jh-footer {
      height: 54px;
      flex-direction: column;
      gap: 3px;
      padding: 4px 10px;
      font-size: 0.74rem;
    }
    .footer-sep {
      display: none;
    }
    .footer-credit {
      max-width: 96vw;
    }
  }
</style>
</head>
<body>
  <!-- 0. 첫 접속 입장 안내창 -->
  <div id="entry-gate">
    <div class="gate-modal">
      <div class="gate-icon">🎄</div>
      <div class="gate-title">입장하시겠습니까?</div>
      <div class="gate-desc">입장하시면 신나는 크리스마스 캐롤과 함께<br>3D 트리가 바로 시작됩니다.<br>(소리를 켜고 입장하세요.)</div>
      <button type="button" class="gate-btn" id="btn-enter">입장하겠습니다</button>
      <div class="gate-warn" id="gate-warn">상위폴더/Music/song.mp3 파일을 찾을 수 없습니다.</div>
    </div>
  </div>

  <!-- 1. 메인 트리 & 솔크 확인 뷰 -->
  <div id="main-view">
    <div id="tree-stage">
      <div id="cv-wrap">
        <canvas id="cv"></canvas>
      </div>
      <div class="hud-bar">
        <span id="hud-left">tree 2026 | pts: 0 | lights: 175</span>
        <span id="hud-right">ready | 60.0 fps</span>
      </div>
    </div>

    <div id="ui-zone">
      <div class="prompt-card" id="prompt-card">
        <div class="card-badge">CHRISTMAS ACCESS CHECK</div>
        <div class="card-title">크리스마스 입장 확인</div>
        <div class="card-desc">올해 크리스마스, 솔크이신가요?<br>정직하게 선택해 주세요.</div>
        <div class="btn-group">
          <button type="button" class="choice-btn btn-solo" id="btn-solo">솔크입니다</button>
          <button type="button" class="choice-btn btn-couple" id="btn-couple">솔크가 아닙니다</button>
        </div>
        <div class="result-banner" id="result-banner">아주 바람직합니다.</div>
      </div>
    </div>

    <div id="solo-toast">아주 바람직합니다.</div>
    <canvas id="fx-canvas"></canvas>
  </div>

  <!-- 2. 404 Not Found 뷰 (배경음악은 끊기지 않고 계속 재생됨) -->
  <div id="page-404">
    <h1>404 Not Found</h1>
    <hr>
    <div class="sub-404">nginx/1.26.2 (Couple Access Denied)</div>
    <div class="couple-msg">커플 저리가</div>
  </div>

  <!-- 3. 최하단 고정 Footer (JH's SW Lab + 유튜브 음원 출처 하이퍼링크) -->
  <footer class="jh-footer">
    <span class="footer-brand">JH's SW Lab</span>
    <span class="footer-sep">|</span>
    <span class="footer-credit">
      음원 출처:
      <a href="https://www.youtube.com/watch?v=grmbM0dDBPk&list=RDgrmbM0dDBPk&start_radio=1"
         target="_blank"
         rel="noopener noreferrer">산타할아버지 우리 마을에(울면 안돼) | 꼬미꼬미 인기동요 | 꼬미꼬미</a>
    </span>
  </footer>

  <script>
  (() => {
    const DATA = __SCENE_JSON__;
    const AUDIO_B64 = "__AUDIO_B64__";

    let hasEntered = false;
    let is404Mode = false;
    let treeAnimId = null;

    function b64ToBytes(b64) {
      const bin = atob(b64);
      const len = bin.length;
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    // ===================================================================
    // 1. 오디오 엔진: 페이지 로드 즉시 메모리 선로딩 + Web Audio 효과음 합성
    // ===================================================================
    let bgm = null;
    let audioCtx = null;
    let gainAttached = false;

    if (AUDIO_B64.length > 0) {
      const mp3Bytes = b64ToBytes(AUDIO_B64);
      const blob = new Blob([mp3Bytes], { type: "audio/mpeg" });
      const blobUrl = URL.createObjectURL(blob);

      bgm = new Audio(blobUrl);
      bgm.loop = true;
      bgm.preload = "auto";
      bgm.playsInline = true;
      bgm.muted = false;
      bgm.volume = 1.0;
      bgm.load(); // 사용자가 '입장하겠습니다'를 누르기 전 미리 디코딩 완료
    } else {
      document.getElementById("gate-warn").style.display = "block";
    }

    function ensureAudioContext() {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      if (!audioCtx) {
        audioCtx = new Ctx();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume().catch(() => {});
      }
      if (bgm && !gainAttached) {
        try {
          const src = audioCtx.createMediaElementSource(bgm);
          const compressor = audioCtx.createDynamicsCompressor();
          compressor.threshold.value = -16;
          compressor.knee.value = 18;
          compressor.ratio.value = 4;
          compressor.attack.value = 0.005;
          compressor.release.value = 0.2;

          const gainNode = audioCtx.createGain();
          gainNode.gain.value = 1.5; // 작은 기기 볼륨에서도 잘 들리도록 1.5배 증폭

          src.connect(compressor);
          compressor.connect(gainNode);
          gainNode.connect(audioCtx.destination);
          gainAttached = true;
        } catch (e) {}
      }
      return audioCtx;
    }

    function keepBgmAlive() {
      if (!bgm) return;
      bgm.muted = false;
      if (bgm.volume < 0.95) bgm.volume = 1.0;
      if (bgm.paused && hasEntered) {
        bgm.play().catch(() => {});
      }
    }

    // 404 페이지에서도 음악이 절대 멈추거나 음소거되지 않도록 상시 유지
    setInterval(() => {
      if (hasEntered) keepBgmAlive();
    }, 450);

    // -------------------------------------------------------------------
    // 효과음 1: '솔크입니다' -> 맑고 영롱한 "뾰로롱~✨" 상승 차임 사운드
    // (배경음악 위에 자연스럽게 믹싱되어 음악이 끊기지 않음)
    // -------------------------------------------------------------------
    function playPyororong() {
      const ctx = ensureAudioContext();
      keepBgmAlive();
      if (!ctx) return;

      const now = ctx.currentTime;
      const notes = [587.33, 783.99, 987.77, 1174.66, 1567.98, 1975.53, 2349.32];

      notes.forEach((freq, i) => {
        const start = now + i * 0.052;
        const dur = 0.36;

        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const g = ctx.createGain();

        osc1.type = "triangle";
        osc1.frequency.setValueAtTime(freq, start);
        osc1.frequency.exponentialRampToValueAtTime(freq * 1.03, start + dur * 0.5);

        osc2.type = "sine";
        osc2.frequency.setValueAtTime(freq * 2.0, start);

        g.gain.setValueAtTime(0.0001, start);
        g.gain.linearRampToValueAtTime(0.26, start + 0.01);
        g.gain.exponentialRampToValueAtTime(0.0008, start + dur);

        const gSub = ctx.createGain();
        gSub.gain.value = 0.35;

        osc1.connect(g);
        osc2.connect(gSub);
        gSub.connect(g);
        g.connect(ctx.destination);

        osc1.start(start);
        osc2.start(start);
        osc1.stop(start + dur + 0.02);
        osc2.stop(start + dur + 0.02);
      });
    }

    // -------------------------------------------------------------------
    // 효과음 2: '솔크가 아닙니다' -> 경쾌하고 날카로운 "삐-비-비-빅!" 경고음
    // (배경음악은 끊기지 않고 404 페이지에서도 계속 재생됨)
    // -------------------------------------------------------------------
    function playPibibibik() {
      const ctx = ensureAudioContext();
      keepBgmAlive();
      if (!ctx) return;

      const now = ctx.currentTime;
      const pulses = [
        { offset: 0.00, dur: 0.09, f1: 1120, f2: 1680 }, // 삐
        { offset: 0.12, dur: 0.08, f1: 1120, f2: 1680 }, // 비
        { offset: 0.23, dur: 0.08, f1: 1120, f2: 1680 }, // 비
        { offset: 0.34, dur: 0.24, f1: 840,  f2: 1260 }, // 빅!
      ];

      pulses.forEach((p) => {
        const st = now + p.offset;
        const oscA = ctx.createOscillator();
        const oscB = ctx.createOscillator();
        const filter = ctx.createBiquadFilter();
        const g = ctx.createGain();

        oscA.type = "square";
        oscA.frequency.setValueAtTime(p.f1, st);

        oscB.type = "sawtooth";
        oscB.frequency.setValueAtTime(p.f2, st);

        filter.type = "lowpass";
        filter.frequency.setValueAtTime(3200, st);

        g.gain.setValueAtTime(0.0001, st);
        g.gain.linearRampToValueAtTime(0.28, st + 0.006);
        g.gain.setValueAtTime(0.26, st + p.dur - 0.018);
        g.gain.exponentialRampToValueAtTime(0.0008, st + p.dur);

        oscA.connect(filter);
        oscB.connect(filter);
        filter.connect(g);
        g.connect(ctx.destination);

        oscA.start(st);
        oscB.start(st);
        oscA.stop(st + p.dur + 0.015);
        oscB.stop(st + p.dur + 0.015);
      });
    }

    // ===================================================================
    // 2. '솔크입니다' 클릭 시 화면 전체에 휘날리는 파티클(컨페티 & 별) 엔진
    // ===================================================================
    const fxCanvas = document.getElementById("fx-canvas");
    const fxCtx = fxCanvas.getContext("2d");
    let particles = [];
    let fxAnimId = null;

    function resizeFxCanvas() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      fxCanvas.width = Math.floor(window.innerWidth * dpr);
      fxCanvas.height = Math.floor(window.innerHeight * dpr);
      fxCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resizeFxCanvas);
    resizeFxCanvas();

    const CONFETTI_COLORS = [
      "#ff3b5c", "#fbbf24", "#34d399", "#60a5fa",
      "#f472b6", "#fde047", "#ffffff", "#a78bfa"
    ];

    function drawStarShape(ctx2d, r) {
      ctx2d.beginPath();
      for (let i = 0; i < 10; i++) {
        const rad = i % 2 === 0 ? r : r * 0.44;
        const a = (i * Math.PI) / 5 - Math.PI / 2;
        const x = Math.cos(a) * rad;
        const y = Math.sin(a) * rad;
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      }
      ctx2d.closePath();
      ctx2d.fill();
    }

    function spawnCelebrationParticles() {
      const W = window.innerWidth;
      const H = window.innerHeight;
      const now = performance.now();

      // 좌·우·중앙 폭죽 + 상단 꽃가루 샤워 총 210개 생성 (약 4.5초간 휘날림)
      for (let i = 0; i < 210; i++) {
        const fromBurst = i < 130;
        let x, y, vx, vy;
        if (fromBurst) {
          const originX = i % 2 === 0 ? W * 0.28 : W * 0.72;
          x = originX + (Math.random() - 0.5) * 80;
          y = H * 0.58 + (Math.random() - 0.5) * 50;
          const angle = -Math.PI * 0.5 + (Math.random() - 0.5) * 1.9;
          const speed = 5.5 + Math.random() * 11.5;
          vx = Math.cos(angle) * speed;
          vy = Math.sin(angle) * speed - 2.5;
        } else {
          x = Math.random() * W;
          y = -20 - Math.random() * H * 0.35;
          vx = (Math.random() - 0.5) * 3.5;
          vy = 2.0 + Math.random() * 4.2;
        }

        particles.push({
          x,
          y,
          vx,
          vy,
          gravity: 0.14 + Math.random() * 0.08,
          drag: 0.985,
          size: 5.5 + Math.random() * 6.5,
          color: CONFETTI_COLORS[(Math.random() * CONFETTI_COLORS.length) | 0],
          kind: i % 5 === 0 ? 1 : (i % 7 === 0 ? 2 : 0), // 0: 종이꽃가루, 1: 별, 2: 빛망울
          rot: Math.random() * Math.PI * 2,
          rotSpd: (Math.random() - 0.5) * 0.22,
          tilt: Math.random() * Math.PI * 2,
          tiltSpd: 0.08 + Math.random() * 0.16,
          swayPh: Math.random() * Math.PI * 2,
          born: now,
          life: 3600 + Math.random() * 1300, // 3.6초 ~ 4.9초 유지
        });
      }

      if (!fxAnimId) {
        fxAnimId = requestAnimationFrame(stepParticles);
      }
    }

    function stepParticles(now) {
      const W = window.innerWidth;
      const H = window.innerHeight;
      fxCtx.clearRect(0, 0, W, H);

      const alive = [];
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const age = now - p.born;
        if (age >= p.life || p.y > H + 60) continue;

        p.vx *= p.drag;
        p.vy = p.vy * p.drag + p.gravity;
        p.x += p.vx + Math.sin(now * 0.004 + p.swayPh) * 0.85;
        p.y += p.vy;
        p.rot += p.rotSpd;
        p.tilt += p.tiltSpd;

        const fadeStart = p.life * 0.68;
        const alpha = age > fadeStart ? Math.max(0, 1 - (age - fadeStart) / (p.life - fadeStart)) : 1;

        fxCtx.save();
        fxCtx.globalAlpha = alpha;
        fxCtx.translate(p.x, p.y);
        fxCtx.rotate(p.rot);
        fxCtx.fillStyle = p.color;

        if (p.kind === 1) {
          drawStarShape(fxCtx, p.size * 0.95);
        } else if (p.kind === 2) {
          fxCtx.shadowColor = p.color;
          fxCtx.shadowBlur = 10;
          fxCtx.beginPath();
          fxCtx.arc(0, 0, p.size * 0.48, 0, Math.PI * 2);
          fxCtx.fill();
        } else {
          const flipScale = Math.cos(p.tilt);
          fxCtx.scale(1, flipScale);
          fxCtx.fillRect(-p.size * 0.6, -p.size * 0.38, p.size * 1.2, p.size * 0.76);
        }
        fxCtx.restore();

        alive.push(p);
      }

      particles = alive;
      if (particles.length > 0 && !is404Mode) {
        fxAnimId = requestAnimationFrame(stepParticles);
      } else {
        fxCtx.clearRect(0, 0, W, H);
        fxAnimId = null;
      }
    }

    // ===================================================================
    // 3. 입장 모달 / 솔크 / 커플 버튼 이벤트 연결
    // ===================================================================
    const entryGate = document.getElementById("entry-gate");
    const btnEnter = document.getElementById("btn-enter");
    const btnSolo = document.getElementById("btn-solo");
    const btnCouple = document.getElementById("btn-couple");
    const resultBanner = document.getElementById("result-banner");
    const soloToast = document.getElementById("solo-toast");
    const mainView = document.getElementById("main-view");
    const page404 = document.getElementById("page-404");

    let tStart = performance.now();
    let toastTimer = null;
    let celebrationBoostUntil = 0;

    function handleEnterSite() {
      if (hasEntered) return;
      hasEntered = true;

      // 사용자 클릭 이벤트 내부에서 즉시 오디오 컨텍스트 및 MP3 재생 (0초 지연)
      ensureAudioContext();
      if (bgm) {
        bgm.muted = false;
        bgm.volume = 1.0;
        bgm.currentTime = 0;
        bgm.play().catch(() => {});
      }

      tStart = performance.now(); // 입장 클릭 순간부터 트리가 바닥에서 자라나기 시작
      entryGate.classList.add("hidden");
    }

    btnEnter.addEventListener("click", handleEnterSite);
    window.addEventListener("keydown", (e) => {
      if (!hasEntered && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        handleEnterSite();
      }
    });

    // '솔크입니다' 클릭: 뾰로롱 효과음 + 아주 바람직합니다 + 파티클 휘날림 (음악 계속 재생)
    btnSolo.addEventListener("click", () => {
      playPyororong();
      spawnCelebrationParticles();
      resultBanner.classList.add("show");
      soloToast.classList.add("visible");
      celebrationBoostUntil = performance.now() + 4200;

      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => {
        soloToast.classList.remove("visible");
      }, 3200);
    });

    // '솔크가 아닙니다' 클릭: 삐비비빅 경고음 + 404 페이지 전환 (음악은 끊기지 않고 계속 재생!)
    btnCouple.addEventListener("click", () => {
      playPibibibik();
      keepBgmAlive();

      is404Mode = true;
      if (treeAnimId) {
        cancelAnimationFrame(treeAnimId);
        treeAnimId = null;
      }

      mainView.style.display = "none";
      page404.classList.add("active");
      document.title = "404 Not Found";

      try {
        window.parent.document.title = "404 Not Found";
        window.parent.history.replaceState({}, "404 Not Found", "/404NotFound");
      } catch (e) {}
    });

    // ===================================================================
    // 4. 실시간 창 크기 대응 + 3D 점자 Z-버퍼 트리 렌더러
    // ===================================================================
    const M = DATA.n;
    const P_i16 = new Int16Array(b64ToBytes(DATA.P).buffer);
    const C_u8  = b64ToBytes(DATA.C);
    const N_i8  = new Int8Array(b64ToBytes(DATA.N).buffer);
    const K_u8  = b64ToBytes(DATA.K);
    const PH_u8 = b64ToBytes(DATA.PH);
    const BD_u16 = new Uint16Array(b64ToBytes(DATA.BD).buffer);

    const N_SNOW = 480;
    const sx0 = new Float32Array(N_SNOW), sy0 = new Float32Array(N_SNOW), sz0 = new Float32Array(N_SNOW);
    const sSpd = new Float32Array(N_SNOW), sPh = new Float32Array(N_SNOW);
    for (let i = 0; i < N_SNOW; i++) {
      sx0[i] = (Math.random() - 0.5) * 3.8;
      sy0[i] = Math.random() * 3.0 - 0.15;
      sz0[i] = (Math.random() - 0.5) * 3.8;
      sSpd[i] = 0.22 + Math.random() * 0.28;
      sPh[i] = Math.random() * Math.PI * 2;
    }

    const cvWrap = document.getElementById("cv-wrap");
    const uiCard = document.getElementById("prompt-card");
    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    const hudL = document.getElementById("hud-left");
    const hudR = document.getElementById("hud-right");

    let CW = 92, CH = 46, PW = 184, PH = 184, W_PX = 552, H_PX = 552;
    let imgData = null, pix = null, bgBuf = null, zbuf = null, cbuf = null;
    let isMobileLayout = window.innerWidth <= 768;

    function updateCanvasGeometry() {
      if (is404Mode) return;
      isMobileLayout = window.innerWidth <= 768;
      const rect = cvWrap.getBoundingClientRect();
      const stageW = Math.max(180, rect.width);
      const stageH = Math.max(220, rect.height);
      const aspect = stageW / stageH;

      const targetCH = isMobileLayout ? 52 : 48;
      const newCH = Math.max(34, Math.min(64, targetCH));
      const newCW = Math.max(36, Math.min(180, Math.round(aspect * newCH * 2)));

      if (newCW === CW && newCH === CH && imgData !== null) return;

      CW = newCW;
      CH = newCH;
      PW = CW * 2;
      PH = CH * 4;
      W_PX = CW * 6;
      H_PX = CH * 12;

      cv.width = W_PX;
      cv.height = H_PX;
      imgData = ctx.createImageData(W_PX, H_PX);
      pix = imgData.data;

      bgBuf = new Uint8ClampedArray(W_PX * H_PX * 4);
      for (let i = 0; i < bgBuf.length; i += 4) {
        bgBuf[i] = 9;
        bgBuf[i + 1] = 13;
        bgBuf[i + 2] = 20;
        bgBuf[i + 3] = 255;
      }

      zbuf = new Float32Array(PW * PH);
      cbuf = new Uint32Array(PW * PH);
    }

    const ro = new ResizeObserver(() => updateCanvasGeometry());
    ro.observe(cvWrap);
    window.addEventListener("resize", updateCanvasGeometry);
    updateCanvasGeometry();

    const INV_P = 1.0 / 4000.0;
    const INV_N = 1.0 / 127.0;
    const PH_SCALE = (Math.PI * 2.0) / 255.0;
    let lastTime = performance.now();
    let fpsSmooth = 60.0;

    function setDot(sx, sy, z, col) {
      if (sx < 0 || sx >= PW || sy < 0 || sy >= PH) return;
      const idx = sy * PW + sx;
      if (z < zbuf[idx]) {
        zbuf[idx] = z;
        cbuf[idx] = col;
      }
    }

    function render(now) {
      if (is404Mode) return;

      const dt = Math.max((now - lastTime) * 0.001, 0.001);
      lastTime = now;
      fpsSmooth = fpsSmooth * 0.92 + (1.0 / dt) * 0.08;

      // 입장 전에는 눈만 내리는 대기 상태(t = 0.25), 입장 클릭 직후 0초부터 트리 빌드업 시작
      const t = hasEntered ? (now - tStart) * 0.001 : 0.25;
      const snowTime = now * 0.001;
      const isCelebrating = now < celebrationBoostUntil;

      zbuf.fill(1e9);
      cbuf.fill(0);

      let usablePH = PH;
      let centerYSub = PH * 0.52;
      if (isMobileLayout && uiCard) {
        const stageRect = cvWrap.getBoundingClientRect();
        const cardRect = uiCard.getBoundingClientRect();
        const overlapRatio = Math.max(
          0.0,
          Math.min(0.36, (stageRect.bottom - cardRect.top + 10) / Math.max(stageRect.height, 1))
        );
        usablePH = PH * (1.0 - overlapRatio * 0.88);
        centerYSub = usablePH * 0.53;
      }

      const camDist = 4.8;
      const focal = Math.min(usablePH * 1.16, PW * 1.82);
      const centerXSub = PW * 0.5;

      const ang = t * (isCelebrating ? 0.85 : 0.52);
      const pitch = 0.16;
      const targetY = 1.20;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);

      let activePts = 0;

      for (let i = 0; i < M; i++) {
        const bd = BD_u16[i] * 0.001;
        if (bd > t) continue;
        activePts++;

        const i3 = i * 3;
        const px0 = P_i16[i3] * INV_P;
        const py0 = P_i16[i3 + 1] * INV_P;
        const pz0 = P_i16[i3 + 2] * INV_P;

        const rx = px0 * ca + pz0 * sa;
        const rz0 = -px0 * sa + pz0 * ca;
        const dy0 = py0 - targetY;
        const cy = targetY + dy0 * cp + rz0 * sp;
        const cz = -dy0 * sp + rz0 * cp;

        const zCam = cz + camDist;
        if (zCam <= 0.6) continue;

        let r = C_u8[i3], g = C_u8[i3 + 1], b = C_u8[i3 + 2];
        const age = t - bd;
        if (age < 0.22) {
          const fl = (1.0 - age / 0.22) * 102.0;
          r += fl; g += fl; b += fl;
        }

        const kind = K_u8[i];
        let glow = false;

        if (kind !== 3 && kind !== 9) {
          const nx0 = N_i8[i3] * INV_N, ny0 = N_i8[i3 + 1] * INV_N, nz0 = N_i8[i3 + 2] * INV_N;
          const nx = nx0 * ca + nz0 * sa;
          const nzt = -nx0 * sa + nz0 * ca;
          const ny = ny0 * cp + nzt * sp;
          const nz = -ny0 * sp + nzt * cp;

          const ndotl = Math.max(0.0, nx * -0.45 + ny * 0.55 + nz * -0.70);
          const atten = Math.min(1.18, Math.max(0.60, 1.08 - 0.15 * cz));
          const diff = (0.50 + 0.60 * ndotl) * atten;
          r *= diff; g *= diff; b *= diff;

          if (kind === 2 || kind === 6 || kind === 4) {
            const ndoth = Math.max(0.0, nx * -0.244 + ny * 0.298 + nz * -0.922);
            const spec = Math.pow(ndoth, 18) * 165.0;
            r += spec; g += spec; b += spec;
          }
        } else if (kind === 3) {
          const ph = PH_u8[i] * PH_SCALE;
          const spd = isCelebrating ? 9.5 : 4.8;
          const tw = 0.35 + 0.65 * Math.pow(0.5 + 0.5 * Math.sin(t * spd + ph), 1.8);
          const mult = (isCelebrating ? 0.78 : 0.55) + 1.15 * tw;
          r *= mult; g *= mult; b *= mult;
          glow = tw > 0.58;
        } else {
          const ph = PH_u8[i] * PH_SCALE;
          const sh = 0.88 + 0.28 * Math.sin(t * 6.5 + ph);
          r *= sh; g *= sh; b *= sh;
          glow = true;
        }

        const ri = r > 255 ? 255 : (r < 0 ? 0 : r | 0);
        const gi = g > 255 ? 255 : (g < 0 ? 0 : g | 0);
        const bi = b > 255 ? 255 : (b < 0 ? 0 : b | 0);
        const packed = (1 << 24) | (ri << 16) | (gi << 8) | bi;

        const invZ = focal / zCam;
        const sx = (centerXSub + rx * invZ) | 0;
        const sy = (centerYSub - (cy - targetY) * invZ) | 0;

        setDot(sx, sy, zCam, packed);
        if (glow) {
          const gCol = (1 << 24) | (((ri * 0.85) | 0) << 16) | (((gi * 0.85) | 0) << 8) | ((bi * 0.85) | 0);
          const gz = zCam + 0.015;
          setDot(sx + 1, sy, gz, gCol);
          setDot(sx - 1, sy, gz, gCol);
          setDot(sx, sy + 1, gz, gCol);
          setDot(sx, sy - 1, gz, gCol);
        }
      }

      // 눈송이 투영
      const snowCol = (1 << 24) | (226 << 16) | (239 << 8) | 255;
      const goldCol = (1 << 24) | (255 << 16) | (225 << 8) | 130;
      const snowSpdMult = isCelebrating ? 1.8 : 1.0;

      for (let i = 0; i < N_SNOW; i++) {
        let wy = (sy0[i] - snowTime * sSpd[i] * snowSpdMult) % 3.0;
        if (wy < 0) wy += 3.0;
        wy -= 0.15;
        const wx = sx0[i] + 0.07 * Math.sin(snowTime * 1.3 + sPh[i]);
        const wz = sz0[i] + 0.07 * Math.cos(snowTime * 1.1 + sPh[i]);
        const sdy = wy - targetY;
        const scy = targetY + sdy * cp + wz * sp;
        const scz = -sdy * sp + wz * cp;
        const zCam = scz + camDist;
        if (zCam > 0.6) {
          const invZ = focal / zCam;
          const col = (isCelebrating && (i % 3 === 0)) ? goldCol : snowCol;
          setDot((centerXSub + wx * invZ) | 0, (centerYSub - (scy - targetY) * invZ) | 0, zCam, col);
        }
      }

      // 2x4 점자 도트 출력
      pix.set(bgBuf);
      for (let sy = 0; sy < PH; sy++) {
        const cyCell = sy >> 2;
        const dy = sy & 3;
        const pyBase = cyCell * 12 + dy * 3 + 1;
        const rowOff = sy * PW;

        for (let sx = 0; sx < PW; sx++) {
          const c = cbuf[rowOff + sx];
          if (c === 0) continue;

          const r = (c >> 16) & 255;
          const g = (c >> 8) & 255;
          const b = c & 255;

          const cxCell = sx >> 1;
          const dx = sx & 1;
          const pxBase = cxCell * 6 + dx * 3 + 1;

          for (let oy = 0; oy < 2; oy++) {
            const rowIdx = (pyBase + oy) * W_PX;
            for (let ox = 0; ox < 2; ox++) {
              const pIdx = (rowIdx + pxBase + ox) * 4;
              pix[pIdx] = r;
              pix[pIdx + 1] = g;
              pix[pIdx + 2] = b;
            }
          }
        }
      }

      ctx.putImageData(imgData, 0, 0);
      hudL.textContent = `tree 2026 | pts: ${activePts.toLocaleString()} | lights: ${DATA.lights}`;
      hudR.textContent = `${bgm && !bgm.paused ? "♪ bgm on | " : ""}${fpsSmooth.toFixed(1)} fps`;

      treeAnimId = requestAnimationFrame(render);
    }

    treeAnimId = requestAnimationFrame(render);
  })();
  </script>
</body>
</html>
"""
    return html.replace("__SCENE_JSON__", scene_json).replace("__AUDIO_B64__", audio_b64)


# =====================================================================
# 4. Streamlit 기본 UI 제거 및 전체 뷰포트 마운트
# =====================================================================
st.markdown(
    """
    <style>
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
    }
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        margin: 0 !important;
        padding: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        overflow: hidden !important;
        background: #070a10 !important;
    }
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
    }
    [data-testid="stMainBlockContainer"] iframe,
    .block-container iframe,
    iframe.stCustomComponentV1 {
        position: fixed !important;
        inset: 0 !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 99999 !important;
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

music_file = find_music_file()
audio_b64_data = (
    load_music_base64(str(music_file), music_file.stat().st_mtime_ns)
    if music_file is not None
    else ""
)

components.html(
    build_responsive_app_html(build_scene_json(), audio_b64_data),
    height=900,
    scrolling=False,
)