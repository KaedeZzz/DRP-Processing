import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
import cv2

from paperdrm import ImagePack, Settings
from paperdrm.drp_direction import drp_direction_map, drp_mask_angle
from paperdrm.spectral_tv import split_text_background
from paperdrm.line_detection import (
        hough_transform,
        find_hough_peaks,
        dominant_orientation_from_accumulator,
        rotate_image_to_orientation,
        overlay_hough_lines,
    )


if __name__ == "__main__":
    def log_stage(msg: str) -> None:
        print(f"[Stage] {msg}")

    log_stage("Loading settings")
    # Centralised settings loaded from YAML with a small override for angle slicing
    settings = Settings.from_yaml("exp_param.yaml").with_overrides(angle_slice=(2, 2), verbose=True)

    log_stage("Loading images + DRP stack")
    # Load images + DRP (2x2 angular slice)
    images = ImagePack(settings=settings)

    log_stage("Computing direction map + mask")
    # Direction map + mask
    mag_map, deg_map = drp_direction_map(images, verbose=settings.verbose)

    # Normalize orientation to 0–255 and show
    target_angle = -60.0  # target direction for DRP
    img = (0.5 * (np.cos(np.radians(deg_map - target_angle)) + 1) * 255).astype(np.uint8)
    plt.imshow(img, cmap="gray")
    plt.title("Direction Map (0–255 normalized)")
    plt.show()

    # log_stage("Running spectral TV decomposition")
    # # Spectral TV decomposition to split text vs paper background
    # img_float = img.astype(np.float32)
    # img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    # text_layer, paper_layer, spectral_result = split_text_background(
    #     img_float,
    #     text_max_time=14,
    #     num_steps=420,
    #     dt=0.06,
    #     eps=2e-6,
    #     verbose=settings.verbose,
    #     band_edges=(0.0, 1.2, 2.4, 4.2, 7.2, 12.0, 25.2),
    #     progress_every=20
    # )

    # corr = img_float / (paper_layer + 1e-6)     
    # ink = np.maximum(1.0 - corr, 0)
    # ink = ink / (np.percentile(ink, 99.5) + 1e-8)
    # ink = np.clip(ink, 0, 1)

    # # def stats(name, arr):
    # #     p = np.percentile(arr, [0, 1, 50, 99, 100])
    # #     print(f"[debug] {name}: min={p[0]:.3e}, p1={p[1]:.3e}, median={p[2]:.3e}, p99={p[3]:.3e}, max={p[4]:.3e}")

    # log_stage("Plotting spectral TV layers")
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].imshow(img_float, cmap="gray")
    # axes[0].set_title("Input")
    # axes[1].imshow(text_layer, cmap="gray")
    # axes[1].set_title("Text (small scale)")
    # axes[2].imshow(paper_layer, cmap="gray")
    # axes[2].set_title("Paper background (large scale)")
    # for ax in axes:
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()

    # d = np.maximum(paper_layer - img_float, 0.0)
    # a = np.percentile(d, 90) / (np.percentile(ink, 90) + 1e-8)
    # paper_only = np.clip(img_float + a * ink, 0.0, 1.0)
    # plt.imshow(paper_only, cmap="gray")
    # img = paper_only
    # plt.title("Ink-subtracted background")
    # plt.show()

    # log_stage("Background subtraction on orientation image")
    # img_float = img.astype(np.float32)
    # bg_sigma = max(3.0, min(img.shape) / 40.0)
    # background = cv2.GaussianBlur(
    #     img_float,
    #     (0, 0),
    #     sigmaX=bg_sigma,
    #     sigmaY=bg_sigma,
    #     borderType=cv2.BORDER_REFLECT101,
    # )
    # residual = img_float - background
    # # limit = np.percentile(np.abs(residual), 99.0)
    # # limit = max(limit, 1.0)
    # # residual = 0.5 + 0.5 * np.tanh(residual / (limit + 1e-6))
    # img = (np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8)
    # plt.imshow(img, cmap="gray")
    # plt.title("Background-subtracted deg_map image")
    # plt.show()

    log_stage("Gabor filter for laid line frequency estimation")

    def _normalize01(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        return (img - mn) / (mx - mn + 1e-8)


    def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate keeping shape."""
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


    def _project_1d(resp: np.ndarray, line_dir_deg: float) -> np.ndarray:
        """
        沿线方向做平均 -> 得到法向 1D 信号 s(n)。
        做法：把线旋到竖直（90°），然后对行方向平均。
        """
        rot = _rotate(resp, angle_deg=(90.0 - line_dir_deg))
        s = rot.mean(axis=0)  # 随“法向”坐标变化
        # 去趋势（高通）：去掉慢变光照/折痕的影响
        win = min(301, max(31, (len(s) // 20) | 1))  # 奇数窗口
        k = np.ones(win, np.float32) / win
        s_hp = s - np.convolve(s, k, mode="same")
        return s_hp.astype(np.float32)


    def estimate_laidline_frequency_gabor(
        img: np.ndarray,
        *,
        line_dir_deg: float,
        periods_px: list[float] | tuple[float, ...] = (8, 10, 12, 14, 16, 18, 22, 26, 32),
        angle_jitter_deg: float = 6.0,
        angle_step_deg: float = 2.0,
        sigma_factor: float = 0.6,
        gamma: float = 0.4,
        ksize: int = 41,
        use_abs_response: bool = True,
    ) -> dict:
        """
        用定向 Gabor 扫描候选周期(=线间距，px)并估计 laid line 频率。

        返回 dict:
        best_period_px, best_freq_cpp, best_theta_deg(法向角), scores{period:score}, best_signal_1d
        """
        img01 = _normalize01(img)

        # Gabor 的 theta 取“法向方向”（振荡方向）
        normal_dir = (line_dir_deg + 90.0) % 180.0
        thetas = np.arange(normal_dir - angle_jitter_deg, normal_dir + angle_jitter_deg + 1e-6, angle_step_deg)
        thetas = (thetas + 180.0) % 180.0

        # 评分：投影后 1D 的“周期能量”
        # 这里用“在目标频率附近的 FFT 能量 / 全频能量”作为分数，更专注于 laid line 周期
        def score_period(signal_1d: np.ndarray, period_px: float) -> float:
            s = signal_1d - signal_1d.mean()
            n = len(s)
            # rfft 频率轴 (cycles/pixel)
            S = np.fft.rfft(s)
            freqs = np.fft.rfftfreq(n, d=1.0)
            f0 = 1.0 / float(period_px)
            # 在 f0±20% 的带内积能量
            band = (freqs >= 0.8 * f0) & (freqs <= 1.2 * f0)
            band_energy = float(np.sum(np.abs(S[band]) ** 2))
            total_energy = float(np.sum(np.abs(S) ** 2)) + 1e-12
            return band_energy / total_energy

        best = {"score": -1.0}
        scores = {}

        for period in periods_px:
            lambd = float(period)  # OpenCV GaborKernel 的 lambd 是像素周期
            sigma = max(1.0, sigma_factor * lambd)
            best_for_period = -1.0
            for th in thetas:
                theta = np.deg2rad(float(th))
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0.0, ktype=cv2.CV_32F)
                kernel -= kernel.mean()  # 更“带通”
                r = cv2.filter2D(img01, cv2.CV_32F, kernel)
                resp = np.abs(r) if use_abs_response else r
                s1d = _project_1d(resp, line_dir_deg=line_dir_deg)
                sc = score_period(s1d, period_px=period)
                if sc > best_for_period:
                    best_for_period = sc
                    if sc > best["score"]:
                        best.update(
                            score=sc,
                            best_period_px=period,
                            best_freq_cpp=1.0 / float(period),
                            best_theta_deg=float(th),
                            best_signal_1d=s1d, # type: ignore
                            best_response=resp, # type: ignore
                        )

            scores[float(period)] = best_for_period

        out = {
            "best_period_px": float(best["best_period_px"]),
            "best_freq_cpp": float(best["best_freq_cpp"]),  # cycles per pixel
            "best_theta_deg": float(best["best_theta_deg"]),  # 法向角（Gabor 用的）
            "scores": scores,
            "best_signal_1d": best["best_signal_1d"],
            "best_response": best["best_response"],
            "line_dir_deg": float(line_dir_deg),
        }
        return out
    
    def rotate_keep_shape(img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate keeping original shape."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )


    def smooth1d(x: np.ndarray, win: int = 31) -> np.ndarray:
        win = int(win)
        if win < 3:
            return x
        if win % 2 == 0:
            win += 1
        k = np.ones(win, np.float32) / win
        return np.convolve(x, k, mode="same")


    def peaks_from_signal(s: np.ndarray, period_px: float) -> np.ndarray:
        """从投影信号中找峰，返回峰的 x 坐标（在‘旋转后’图像的列坐标系里）。"""
        s = np.asarray(s, np.float32)
        s = s - np.median(s)
        s = smooth1d(s, win=max(9, int(period_px * 0.7) | 1))

        # laid lines 间距约等于 period_px，所以峰的最小距离可设为 ~0.7*period
        min_dist = max(3, int(period_px * 0.7))

        # 简易峰：局部最大 + 距离筛选
        idx = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1
        if len(idx) == 0:
            return idx.astype(int)
        keep = [int(idx[0])]
        for p in idx[1:]:
            if int(p) - keep[-1] >= min_dist:
                keep.append(int(p))
        return np.array(keep, dtype=int)


    def overlay_laid_lines(
        img: np.ndarray,
        *,
        line_dir_deg: float,
        best_signal_1d: np.ndarray,
        best_period_px: float,
        color=(0, 0, 255),   # BGR: 红色
        thickness: int = 1,
        alpha: float = 0.6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        返回：
        overlay: 在原图上叠加了 laid lines 的彩色图
        peaks_x: 峰位置（在旋转后的图像坐标系里）
        """
        # 保证 img 是 3 通道方便画彩色线
        if img.ndim == 2:
            base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            base = img.copy()

        h, w = base.shape[:2]

        # 1) 旋转：把“线方向”旋到竖直（90°）
        rot_angle = 90.0 - float(line_dir_deg)
        base_rot = rotate_keep_shape(base, rot_angle)

        # 2) 找峰：峰位置就是 laid line 在旋转图里的 x
        peaks_x = peaks_from_signal(best_signal_1d, best_period_px)

        # 3) 在旋转图上画竖线
        overlay_rot = base_rot.copy()
        for x in peaks_x:
            x = int(np.clip(x, 0, w - 1))
            cv2.line(overlay_rot, (x, 0), (x, h - 1), color, thickness, lineType=cv2.LINE_AA)

        # 4) 旋回去并 alpha blend
        overlay_back = rotate_keep_shape(overlay_rot, -rot_angle)

        overlay = cv2.addWeighted(base, 1.0 - alpha, overlay_back, alpha, 0.0)
        return overlay, peaks_x
    
    out = estimate_laidline_frequency_gabor(img, line_dir_deg=90.0,
                                           periods_px=list(range(6, 41, 2)))
    print("dominant orientation: ", out["line_dir_deg"], " best period(px) =", out["best_period_px"], " best freq =", out["best_freq_cpp"], "cpp")
    overlay, peaks_x = overlay_laid_lines(
        img,
        line_dir_deg=out["line_dir_deg"],
        best_signal_1d=out["best_signal_1d"],
        best_period_px=out["best_period_px"],
        color=(0, 0, 255),
        thickness=1,
        alpha=0.65
    )
    cv2.imwrite("laid_lines_overlay.png", overlay)
    cv2.imshow("Laid Lines Overlay", overlay)
    win = "Laid Lines Overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # 允许调整大小
    cv2.resizeWindow(win, 1200, 800)          # 设置窗口宽高（像素）
    cv2.imshow(win, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # log_stage("Computing column intensity profile")
    # # Column intensity profile
    # img_stacked = img.mean(axis=0)
    # plt.plot(img_stacked)
    # plt.title("Stacked Image Intensity Profile")
    # plt.xlabel("Pixel Position")
    # plt.ylabel("Average Intensity")
    # plt.show()

    # log_stage("Peak detection and overlay")
    # # Peak detection (top 13 by height) and overlay
    # peaks, _ = scipy.signal.find_peaks(img_stacked)
    # peaks_by_height = peaks[np.argsort(img_stacked[peaks])[::-1]]
    # keep = 50
    # plt.imshow(img, cmap="gray")
    # for peak in peaks_by_height[:keep]:
    #     plt.plot([peak, peak], [0, img.shape[0]], color="red", linewidth=1)
    # plt.show()
