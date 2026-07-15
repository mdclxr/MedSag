import logging
import math
import os
import sys
from typing import Iterable

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _ensure_medsam2_repo(repo_path: str):
    if not repo_path or not os.path.isdir(repo_path):
        raise FileNotFoundError(f"MedSAM2 repo path does not exist: {repo_path}")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _resolve_config_argument(repo_path: str, config_path: str) -> str:
    abs_cfg = os.path.abspath(config_path)
    sam2_cfg_root = os.path.join(os.path.abspath(repo_path), 'sam2', 'configs')
    cfg_root = os.path.join(os.path.abspath(repo_path), 'efficient_track_anything', 'configs')
    if abs_cfg.startswith(sam2_cfg_root + os.sep):
        return os.path.join('configs', os.path.relpath(abs_cfg, sam2_cfg_root))
    if abs_cfg.startswith(cfg_root + os.sep):
        return os.path.join('configs', os.path.relpath(abs_cfg, cfg_root))
    if os.path.exists(abs_cfg):
        return abs_cfg
    return config_path


def _window_ct(image_hu: np.ndarray, ww: float, wl: float) -> np.ndarray:
    low = float(wl) - float(ww) / 2.0
    high = float(wl) + float(ww) / 2.0
    image = np.clip(image_hu.astype(np.float32), low, high)
    denom = max(high - low, 1e-6)
    image = (image - low) / denom * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _resize_grayscale_stack(array_zyx: np.ndarray, image_size: int) -> np.ndarray:
    from PIL import Image

    depth, _, _ = array_zyx.shape
    resized = np.zeros((depth, 3, image_size, image_size), dtype=np.float32)
    for idx in range(depth):
        pil = Image.fromarray(array_zyx[idx].astype(np.uint8), mode='L').convert('RGB')
        pil = pil.resize((image_size, image_size))
        resized[idx] = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)
    return resized


def _prepare_volume_tensor(image_zyx: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    if image_zyx.shape[1] != image_size or image_zyx.shape[2] != image_size:
        frames = _resize_grayscale_stack(image_zyx, image_size)
    else:
        frames = image_zyx[:, None].repeat(3, axis=1).astype(np.float32)

    frames = frames / 255.0
    tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    tensor = (tensor - mean) / std
    return tensor


def _clamp_point(point: Iterable[float], width: int, height: int, depth: int) -> tuple[int, int, int, int]:
    x, y, z, label = point
    cx = max(0, min(int(round(float(x))), width - 1))
    cy = max(0, min(int(round(float(y))), height - 1))
    cz = max(0, min(int(round(float(z))), depth - 1))
    cl = 1 if int(label) > 0 else 0
    return cx, cy, cz, cl


def _clamp_box(box: Iterable[float], width: int, height: int, depth: int) -> tuple[int, int, int, int, int, int]:
    x1, y1, z1, x2, y2, z2 = box
    x_min = max(0, min(int(round(min(x1, x2))), width - 1))
    y_min = max(0, min(int(round(min(y1, y2))), height - 1))
    z_min = max(0, min(int(round(min(z1, z2))), depth - 1))
    x_max = max(0, min(int(round(max(x1, x2))), width - 1))
    y_max = max(0, min(int(round(max(y1, y2))), height - 1))
    z_max = max(0, min(int(round(max(z1, z2))), depth - 1))
    return x_min, y_min, z_min, x_max, y_max, z_max


def _build_seed_mask_from_points(points_xy: list[tuple[int, int]], radius_px: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    radius_px = max(2, int(radius_px))
    yy, xx = np.ogrid[:height, :width]
    for px, py in points_xy:
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        mask[dist2 <= radius_px ** 2] = 1
    return mask


def _extract_2d_component_near_seed(mask_2d: np.ndarray, seed_xy: tuple[int, int] | None = None) -> np.ndarray:
    from scipy import ndimage

    binary = mask_2d.astype(bool)
    if not np.any(binary):
        return mask_2d.astype(np.uint8)

    structure = ndimage.generate_binary_structure(2, 2)
    labeled, num = ndimage.label(binary, structure=structure)
    if num <= 1:
        return binary.astype(np.uint8)

    component_id = 0
    if seed_xy is not None:
        sx, sy = seed_xy
        if 0 <= sy < labeled.shape[0] and 0 <= sx < labeled.shape[1]:
            component_id = int(labeled[sy, sx])

    component_ids = np.arange(1, num + 1, dtype=np.int32)
    if component_id <= 0:
        if seed_xy is not None:
            sx, sy = seed_xy
            best_component = 0
            best_distance = None
            for cid in component_ids:
                coords = np.argwhere(labeled == int(cid))
                if coords.size == 0:
                    continue
                center_y, center_x = coords.mean(axis=0)
                dist = ((center_x - sx) ** 2 + (center_y - sy) ** 2) ** 0.5
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_component = int(cid)
            component_id = best_component
        if component_id <= 0:
            sizes = ndimage.sum(binary, labeled, index=component_ids)
            if np.isscalar(sizes):
                sizes = np.array([sizes], dtype=np.float64)
            component_id = int(component_ids[int(np.argmax(sizes))])

    filtered = np.zeros_like(mask_2d, dtype=np.uint8)
    filtered[labeled == component_id] = 1
    return filtered


def _compute_prompt_mask_stats(prompt_mask_2d: np.ndarray, spacing_xyz: tuple[float, float, float]) -> dict:
    coords = np.argwhere(prompt_mask_2d > 0)
    if coords.size == 0:
        return {
            'seed_xy': None,
            'area_px': 0,
            'bbox_w_px': 0,
            'bbox_h_px': 0,
            'diameter_mm': 0.0,
            'max_slice_span': 0,
        }

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    bbox_w_px = int(max_x - min_x + 1)
    bbox_h_px = int(max_y - min_y + 1)
    cy, cx = coords.mean(axis=0)
    sx, sy, sz = spacing_xyz
    diameter_mm = max(bbox_w_px * float(sx), bbox_h_px * float(sy))
    max_slice_span = max(3, int(math.ceil((diameter_mm / max(float(sz), 1e-6)) * 2.2)))
    max_slice_span = min(max_slice_span, 24)
    return {
        'seed_xy': (int(round(cx)), int(round(cy))),
        'area_px': int(coords.shape[0]),
        'bbox_w_px': bbox_w_px,
        'bbox_h_px': bbox_h_px,
        'diameter_mm': float(diameter_mm),
        'max_slice_span': max_slice_span,
    }


def _extract_seed_component(
    segs_3d: np.ndarray,
    seed_xyz: tuple[int, int, int] | None,
    label_value: int,
) -> np.ndarray:
    from scipy import ndimage

    target = segs_3d == int(label_value)
    if not np.any(target):
        return segs_3d

    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(target, structure=structure)
    if num <= 1:
        return segs_3d

    component_id = 0
    if seed_xyz is not None:
        sx, sy, sz = seed_xyz
        if 0 <= sz < labeled.shape[0] and 0 <= sy < labeled.shape[1] and 0 <= sx < labeled.shape[2]:
            component_id = int(labeled[sz, sy, sx])

    if component_id <= 0:
        component_ids = np.arange(1, num + 1, dtype=np.int32)
        sizes = ndimage.sum(target, labeled, index=component_ids)
        if np.isscalar(sizes):
            sizes = np.array([sizes], dtype=np.float64)
        if seed_xyz is not None:
            sx, sy, sz = seed_xyz
            best_component = 0
            best_distance = None
            for cid in component_ids:
                coords = np.argwhere(labeled == int(cid))
                if coords.size == 0:
                    continue
                center_z, center_y, center_x = coords.mean(axis=0)
                dist = ((center_x - sx) ** 2 + (center_y - sy) ** 2 + (center_z - sz) ** 2) ** 0.5
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_component = int(cid)
            component_id = best_component
        if component_id <= 0:
            component_id = int(component_ids[int(np.argmax(sizes))])

    filtered = np.zeros_like(segs_3d, dtype=np.uint8)
    filtered[labeled == component_id] = int(label_value)
    return filtered


def _regularize_small_nodule_volume(
    segs_3d: np.ndarray,
    label_value: int,
    seed_xyz: tuple[int, int, int] | None,
    prompt_stats: dict | None = None,
) -> np.ndarray:
    from scipy import ndimage

    mask = segs_3d == int(label_value)
    if not np.any(mask):
        return segs_3d

    structure3d = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(mask, structure=structure3d, iterations=1)
    filled = ndimage.binary_fill_holes(closed)

    if seed_xyz is not None:
        sx, sy, sz = seed_xyz
        if 0 <= sz < filled.shape[0] and 0 <= sy < filled.shape[1] and 0 <= sx < filled.shape[2]:
            if not filled[sz, sy, sx]:
                filled[sz, sy, sx] = True

    max_span = int((prompt_stats or {}).get('max_slice_span') or 0)
    if seed_xyz is not None and max_span > 0:
        _, _, sz = seed_xyz
        radius = max(1, int(math.ceil(max_span / 2.0)))
        z_min = max(0, sz - radius)
        z_max = min(filled.shape[0], sz + radius + 1)
        limited = np.zeros_like(filled, dtype=bool)
        limited[z_min:z_max] = filled[z_min:z_max]
        filled = limited

    regularized = np.zeros_like(segs_3d, dtype=np.uint8)
    regularized[filled] = int(label_value)
    return regularized


def _validate_segmentation_extent(
    segs_3d: np.ndarray,
    label_value: int,
    seed_slice_idx: int | None = None,
    prompt_stats: dict | None = None,
) -> None:
    mask = segs_3d == int(label_value)
    voxels = int(mask.sum())
    if voxels <= 0:
        raise ValueError("模型未生成有效掩码，请重新点击更准确的病灶中心")

    total_voxels = int(mask.size)
    volume_ratio = voxels / max(total_voxels, 1)
    slice_counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    active_slices = int(np.count_nonzero(slice_counts))
    max_slice_area = int(slice_counts.max()) if slice_counts.size else 0
    slice_area_ratio = max_slice_area / max(mask.shape[1] * mask.shape[2], 1)
    active_slice_indices = np.where(slice_counts > 0)[0]

    if volume_ratio > 0.12:
        raise ValueError("推理结果范围异常偏大，已拦截本次结果，请重新点击更精确的病灶中心")
    dynamic_max_span = None
    if prompt_stats:
        dynamic_max_span = int(prompt_stats.get('max_slice_span') or 0)
    if seed_slice_idx is not None and active_slice_indices.size and dynamic_max_span:
        min_slice = int(active_slice_indices.min())
        max_slice = int(active_slice_indices.max())
        actual_span = max_slice - min_slice + 1
        if actual_span > dynamic_max_span:
            span_ratio = actual_span / max(dynamic_max_span, 1)
            # Allow larger z-span when the single-slice area and whole-volume ratio
            # still look anatomically plausible for a larger nodule.
            if span_ratio > 2.2 and (slice_area_ratio > 0.08 or volume_ratio > 0.015):
                raise ValueError(
                    f"推理结果跨层过多（{actual_span} 层），且整体范围明显偏大，已拦截本次结果；当前粗标目标预计不应超过 {dynamic_max_span} 层"
                )
    if active_slices > max(24, int(mask.shape[0] * 0.18)):
        # Large nodules can legitimately span more slices, so only reject here
        # when the z-span is accompanied by an obviously overgrown footprint.
        if slice_area_ratio > 0.1 or volume_ratio > 0.02:
            raise ValueError("推理结果跨层过多且范围异常偏大，已拦截本次结果，请重新点击病灶中心")
    if slice_area_ratio > 0.35:
        raise ValueError("推理结果单层面积异常偏大，已拦截本次结果，请重新点击病灶中心")


class MedSAM2GPUService:
    def __init__(self, repo_path: str, checkpoint_path: str, config_path: str, device: str = 'cuda:0'):
        self.repo_path = os.path.abspath(repo_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config_path = os.path.abspath(config_path)
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._predictor = None
        self._image_size = 512
        self._load_model()

    def _load_model(self):
        _ensure_medsam2_repo(self.repo_path)
        config_arg = _resolve_config_argument(self.repo_path, self.config_path)
        use_sam2 = (
            os.path.sep + 'sam2' + os.path.sep in self.config_path
            or config_arg.startswith('sam2')
        )
        if use_sam2:
            from sam2.build_sam import build_sam2_video_predictor_npz
            builder = build_sam2_video_predictor_npz
        else:
            from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor_npz
            builder = build_efficienttam_video_predictor_npz
        logger.info(
            "Loading MedSAM2 predictor repo=%s ckpt=%s cfg=%s device=%s",
            self.repo_path,
            self.checkpoint_path,
            config_arg,
            self.device,
        )
        self._predictor = builder(
            config_arg,
            self.checkpoint_path,
            device=str(self.device),
        )
        self._predictor.eval()

    def health(self) -> dict:
        return {
            'ready': self._predictor is not None,
            'device': str(self.device),
            'checkpoint': self.checkpoint_path,
            'config': self.config_path,
            'gpu_available': torch.cuda.is_available(),
        }

    @torch.inference_mode()
    def segment_volume(
        self,
        volume_path: str,
        output_path: str,
        points: list[list[float]] | None = None,
        boxes: list[list[float]] | None = None,
        prompt_mask_path: str | None = None,
        prompt_mask_axis: str = 'z',
        prompt_mask_slice_idx: int | None = None,
        prompt_mask_label: int | None = None,
        label_value: int = 1,
        ww: float = 1500.0,
        wl: float = -600.0,
    ) -> str:
        import SimpleITK as sitk

        if not points and not boxes and not prompt_mask_path:
            raise ValueError("At least one point, one box, or one mask prompt is required")

        image = sitk.ReadImage(volume_path)
        image_zyx = sitk.GetArrayFromImage(image)
        depth, height, width = image_zyx.shape
        image_8bit = _window_ct(image_zyx, ww=ww, wl=wl)
        image_tensor = _prepare_volume_tensor(image_8bit, self._image_size, self.device)

        segs_3d = np.zeros((depth, height, width), dtype=np.uint8)
        point_groups: dict[int, list[tuple[int, int, int]]] = {}
        prompt_slices: set[int] = set()
        positive_seed_xyz: tuple[int, int, int] | None = None
        prompt_masks_by_slice: dict[int, np.ndarray] = {}
        prompt_mask_stats: dict | None = None
        seed_slice_idx: int | None = None

        for point in points or []:
            px, py, pz, pl = _clamp_point(point, width, height, depth)
            prompt_slices.add(pz)
            point_groups.setdefault(pz, []).append((px, py, pl))
            if pl > 0 and positive_seed_xyz is None:
                positive_seed_xyz = (px, py, pz)

        box_groups: dict[int, list[tuple[int, int, int, int]]] = {}
        for box in boxes or []:
            x_min, y_min, z_min, x_max, y_max, z_max = _clamp_box(box, width, height, depth)
            z_mid = int(round((z_min + z_max) / 2.0))
            prompt_slices.add(z_mid)
            box_groups.setdefault(z_mid, []).append((x_min, y_min, x_max, y_max))

        if prompt_mask_path:
            if prompt_mask_axis != 'z':
                raise ValueError("当前仅支持使用轴状面手工掩码作为 AI 推理提示")
            if prompt_mask_slice_idx is None:
                raise ValueError("缺少手工掩码切片索引")

            prompt_mask_img = sitk.ReadImage(prompt_mask_path)
            prompt_mask_arr = sitk.GetArrayFromImage(prompt_mask_img)
            if prompt_mask_arr.shape != image_zyx.shape:
                raise ValueError(
                    f'提示掩码尺寸与 CT 体数据不一致: mask={prompt_mask_arr.shape}, volume={image_zyx.shape}'
                )

            slice_idx = max(0, min(int(prompt_mask_slice_idx), depth - 1))
            mask_label = int(prompt_mask_label or label_value)
            prompt_mask = (prompt_mask_arr[slice_idx] == mask_label).astype(np.uint8)
            if not np.any(prompt_mask):
                raise ValueError("当前轴状面切片还没有对应类别的手工掩码，请先粗标后再执行 AI 标注")

            prompt_mask_stats = _compute_prompt_mask_stats(prompt_mask, image.GetSpacing())
            prompt_mask = _extract_2d_component_near_seed(prompt_mask, prompt_mask_stats.get('seed_xy'))
            prompt_mask_stats = _compute_prompt_mask_stats(prompt_mask, image.GetSpacing())
            prompt_masks_by_slice[slice_idx] = prompt_mask
            prompt_slices.add(slice_idx)
            seed_slice_idx = slice_idx
            coords = np.argwhere(prompt_mask > 0)
            if coords.size > 0:
                cy, cx = coords.mean(axis=0)
                positive_seed_xyz = (int(round(cx)), int(round(cy)), slice_idx)

        if not prompt_slices:
            raise ValueError("Prompt slices could not be determined")

        sorted_prompt_slices = sorted(prompt_slices)

        def _propagate_from_mask(prompt_mask: np.ndarray, slice_idx: int):
            inference_state = self._predictor.init_state(image_tensor, height, width)
            _, _, masks = self._predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                mask=prompt_mask,
            )
            segs_3d[slice_idx, (masks[0] > 0.0).detach().cpu().numpy()[0]] = int(label_value)

            try:
                iterator = self._predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=slice_idx,
                    reverse=False,
                )
            except TypeError:
                iterator = self._predictor.propagate_in_video(inference_state)
            for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                for mask_idx, _ in enumerate(out_obj_ids):
                    mask = (out_mask_logits[mask_idx] > 0.0).detach().cpu().numpy()[0]
                    segs_3d[out_frame_idx, mask] = int(label_value)

            self._predictor.reset_state(inference_state)
            inference_state = self._predictor.init_state(image_tensor, height, width)
            self._predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                mask=prompt_mask,
            )
            try:
                iterator = self._predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=slice_idx,
                    reverse=True,
                )
            except TypeError:
                iterator = ()
            for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                for mask_idx, _ in enumerate(out_obj_ids):
                    mask = (out_mask_logits[mask_idx] > 0.0).detach().cpu().numpy()[0]
                    segs_3d[out_frame_idx, mask] = int(label_value)
            self._predictor.reset_state(inference_state)

        for slice_idx in sorted_prompt_slices:
            if slice_idx in prompt_masks_by_slice:
                _propagate_from_mask(prompt_masks_by_slice[slice_idx], slice_idx)
                continue

            if slice_idx in box_groups:
                merged_box = box_groups[slice_idx][0]
                if len(box_groups[slice_idx]) > 1:
                    xs = [item[0] for item in box_groups[slice_idx]] + [item[2] for item in box_groups[slice_idx]]
                    ys = [item[1] for item in box_groups[slice_idx]] + [item[3] for item in box_groups[slice_idx]]
                    merged_box = (min(xs), min(ys), max(xs), max(ys))
                inference_state = self._predictor.init_state(image_tensor, height, width)
                _, _, mask_logits = self._predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=slice_idx,
                    obj_id=1,
                    box=np.array(merged_box, dtype=np.float32),
                )
                prompt_mask = (mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                self._predictor.reset_state(inference_state)
                _propagate_from_mask(prompt_mask, slice_idx)
                continue

            slice_points = point_groups[slice_idx]
            point_coords = np.array([[px, py] for px, py, _ in slice_points], dtype=np.float32)
            point_labels = np.array([1 if pl > 0 else 0 for _, _, pl in slice_points], dtype=np.int32)
            if not np.any(point_labels == 1):
                raise ValueError(f"Slice {slice_idx} does not contain positive points")

            inference_state = self._predictor.init_state(image_tensor, height, width)
            _, _, mask_logits = self._predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
            )
            prompt_mask = (mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            self._predictor.reset_state(inference_state)

            # Fallback if the native point prompt produces an empty mask.
            if not np.any(prompt_mask):
                positive_points = [(px, py) for px, py, pl in slice_points if pl > 0]
                negative_points = [(px, py) for px, py, pl in slice_points if pl == 0]
                radius_px = max(4, int(round(math.sqrt(height * width) * 0.015)))
                prompt_mask = _build_seed_mask_from_points(positive_points, radius_px, height, width)
                for nx, ny in negative_points:
                    rr, cc = np.ogrid[:height, :width]
                    prompt_mask[(cc - nx) ** 2 + (rr - ny) ** 2 <= radius_px ** 2] = 0

            _propagate_from_mask(prompt_mask, slice_idx)

        segs_3d = _extract_seed_component(segs_3d, positive_seed_xyz, label_value)
        segs_3d = _regularize_small_nodule_volume(
            segs_3d,
            label_value,
            positive_seed_xyz,
            prompt_stats=prompt_mask_stats,
        )
        _validate_segmentation_extent(
            segs_3d,
            label_value,
            seed_slice_idx=seed_slice_idx,
            prompt_stats=prompt_mask_stats,
        )

        out_img = sitk.GetImageFromArray(segs_3d.astype(np.uint8))
        out_img.SetSpacing(image.GetSpacing())
        out_img.SetOrigin(image.GetOrigin())
        out_img.SetDirection(image.GetDirection())
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(out_img, output_path)
        return output_path
