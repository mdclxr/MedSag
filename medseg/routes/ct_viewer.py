"""
CT 体数据查看器蓝图
提供切片 API 和 CT 项目浏览页面
支持 MHD/RAW 和 NIfTI (.nii / .nii.gz) 分割掩码可视化
"""

import logging
import os
import shutil
import csv
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO, StringIO
import zipfile
from flask import Blueprint, current_app, render_template, request, jsonify, Response, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import requests
from medseg.config import PROJECTS_FOLDER, MEDSAM2_SERVICE_TIMEOUT, MEDSAM2_SERVICE_URL
from medseg.models.project import Project
from medseg.models.user import get_user_assigned_images, update_image_status

logger = logging.getLogger(__name__)

bp = Blueprint('ct_viewer', __name__)
_medsam2_task_cache: dict[str, dict] = {}
_medsam2_task_lock = threading.Lock()


def _projects_folder() -> str:
    return current_app.config.get('PROJECTS_FOLDER', PROJECTS_FOLDER)


def _medsam2_service_url() -> str:
    return current_app.config.get('MEDSAM2_SERVICE_URL', MEDSAM2_SERVICE_URL).rstrip('/')


def _medsam2_service_timeout() -> int:
    return int(current_app.config.get('MEDSAM2_SERVICE_TIMEOUT', MEDSAM2_SERVICE_TIMEOUT))


def _is_project_local_path(project_path: str, candidate_path: str) -> bool:
    if not project_path or not candidate_path:
        return False
    try:
        common = os.path.commonpath([os.path.abspath(project_path), os.path.abspath(candidate_path)])
        return common == os.path.abspath(project_path)
    except Exception:
        return False


def _build_ct_mask_storage_path(project_path: str, volume_name: str, volume_id: int, source_path: str) -> str:
    masks_dir = os.path.join(project_path, 'ct_volumes', 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    src_lower = str(source_path or '').lower()
    if src_lower.endswith('.nii.gz'):
        ext = '.nii.gz'
    elif src_lower.endswith('.nii'):
        ext = '.nii'
    else:
        ext = '.nii.gz'

    stem = os.path.splitext(os.path.splitext(os.path.basename(volume_name or 'ct_volume'))[0])[0]
    safe_stem = secure_filename(stem) or f'ct_volume_{volume_id}'
    return os.path.join(masks_dir, f'{safe_stem}_mask_{volume_id}{ext}')


def _ensure_project_managed_mask(project: Project, volume_id: int, preferred_nii_path: str = None):
    volume = project.get_ct_volume(volume_id)
    if not volume:
        raise FileNotFoundError('CT 体数据不存在')

    candidate_path = (preferred_nii_path or volume.get('nii_path') or '').strip()
    if not candidate_path:
        raise FileNotFoundError('当前体数据尚未初始化掩码文件')
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(f'掩码文件不存在: {candidate_path}')

    project_path = project.project_path
    if _is_project_local_path(project_path, candidate_path):
        return volume, os.path.abspath(candidate_path), False

    target_path = _build_ct_mask_storage_path(
        project_path,
        volume.get('name') or f'ct_volume_{volume_id}',
        volume_id,
        candidate_path,
    )

    if os.path.abspath(candidate_path) != os.path.abspath(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(candidate_path, target_path)

    project.update_ct_volume_nii(volume_id, target_path)
    volume['nii_path'] = target_path
    return volume, os.path.abspath(target_path), True


def _merge_ct_masks(base_mask_path: str, new_mask_path: str, output_path: str | None = None) -> str:
    import SimpleITK as sitk
    from medseg.utils.ct_utils import clear_volume_cache

    base_img = sitk.ReadImage(base_mask_path)
    new_img = sitk.ReadImage(new_mask_path)

    base_arr = sitk.GetArrayFromImage(base_img).astype('uint16')
    new_arr = sitk.GetArrayFromImage(new_img).astype('uint16')

    if base_arr.shape != new_arr.shape:
        raise ValueError(
            f'掩码尺寸不一致，无法合并: base={base_arr.shape}, new={new_arr.shape}'
        )

    merged = base_arr.copy()
    overwrite_mask = new_arr > 0
    merged[overwrite_mask] = new_arr[overwrite_mask]

    out_path = output_path or base_mask_path
    merged_img = sitk.GetImageFromArray(merged.astype('uint16'))
    merged_img.SetSpacing(base_img.GetSpacing())
    merged_img.SetOrigin(base_img.GetOrigin())
    merged_img.SetDirection(base_img.GetDirection())
    sitk.WriteImage(merged_img, out_path)
    clear_volume_cache(base_mask_path)
    clear_volume_cache(new_mask_path)
    if out_path != base_mask_path:
        clear_volume_cache(out_path)
    return out_path


def _build_ct_undo_backup_path(project_path: str, volume_name: str, volume_id: int) -> str:
    undo_dir = os.path.join(project_path, 'ct_volumes', 'masks', '.undo')
    os.makedirs(undo_dir, exist_ok=True)
    stem = os.path.splitext(os.path.splitext(os.path.basename(volume_name or 'ct_volume'))[0])[0]
    safe_stem = secure_filename(stem) or f'ct_volume_{volume_id}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(undo_dir, f'{safe_stem}_undo_{volume_id}_{timestamp}_{uuid.uuid4().hex[:8]}.nii.gz')


def _build_ct_init_mask_path(project_path: str, volume_name: str, volume_id: int) -> str:
    masks_dir = os.path.join(project_path, 'ct_volumes', 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    stem = os.path.splitext(os.path.splitext(os.path.basename(volume_name or 'ct_volume'))[0])[0]
    safe_stem = secure_filename(stem) or f'ct_volume_{volume_id}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(masks_dir, f'{safe_stem}_init_{volume_id}_{timestamp}_{uuid.uuid4().hex[:8]}.nii.gz')


def _create_empty_mask_like(reference_mask_path: str, output_path: str) -> str:
    import SimpleITK as sitk

    ref_img = sitk.ReadImage(reference_mask_path)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    empty_img = sitk.GetImageFromArray((ref_arr * 0).astype('uint16'))
    empty_img.CopyInformation(ref_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(empty_img, output_path)
    return output_path


def initialize_ct_volume_mask(project: Project, volume_id: int, action: str = 'create_blank') -> dict:
    """
    为 CT 体数据自动创建或重建关联掩码。
    """
    action = str(action or 'create_blank').strip().lower()
    if action not in {'create_blank', 'create_from_csv', 'replace_from_csv'}:
        raise ValueError('不支持的初始化动作')

    volume = project.get_ct_volume(volume_id)
    if not volume:
        raise FileNotFoundError('CT 体数据不存在')

    vol_path = project.get_ct_volume_path(volume_id)
    if not vol_path or not os.path.isfile(vol_path):
        raise FileNotFoundError('CT 体数据文件不存在')

    existing_nii_path = str(volume.get('nii_path') or '').strip()
    existing_mask_exists = bool(existing_nii_path and os.path.isfile(existing_nii_path))
    annotations = project.get_ct_annotations_for_volume(volume_id)
    has_csv_annotations = bool(annotations)

    if action in {'create_blank', 'create_from_csv'} and existing_mask_exists:
        raise FileExistsError('当前已存在掩码，请直接继续编辑现有掩码')
    if action in {'create_from_csv', 'replace_from_csv'} and not has_csv_annotations:
        raise ValueError('当前没有可用的 CSV 标注数据，无法根据 CSV 生成初始掩码')

    import SimpleITK as sitk
    import numpy as np
    from skimage.draw import disk

    image = sitk.ReadImage(vol_path)
    shape = image.GetSize()[::-1]
    mask_arr = np.zeros(shape, dtype=np.uint8)
    project_classes = project.get_classes() or ['nodule']
    class_to_value = {str(name).strip().lower(): idx + 1 for idx, name in enumerate(project_classes)}
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    if action in {'create_from_csv', 'replace_from_csv'}:
        for anno in annotations:
            world_x, world_y, world_z = anno['coord_x'], anno['coord_y'], anno['coord_z']
            diam = anno.get('diameter_mm', 0)
            label_name = str(anno.get('label') or '').strip().lower()
            label_value = class_to_value.get(label_name, 1)
            if diam <= 0:
                diam = 10.0

            radius_mm = diam / 2.0
            vx = int(round((world_x - origin[0]) / spacing[0]))
            vy = int(round((world_y - origin[1]) / spacing[1]))
            vz = int(round((world_z - origin[2]) / spacing[2]))
            z_radius = int(round(radius_mm / spacing[2]))

            for z in range(max(0, vz - z_radius), min(shape[0], vz + z_radius + 1)):
                dz = (z - vz) * spacing[2]
                r_plane = np.sqrt(max(0, radius_mm ** 2 - dz ** 2))
                rx_pix = r_plane / spacing[0]
                ry_pix = r_plane / spacing[1]
                r_pix = (rx_pix + ry_pix) / 2.0
                if r_pix < 0.5:
                    r_pix = 1.0
                rr, cc_img = disk((vy, vx), r_pix, shape=(shape[1], shape[2]))
                mask_arr[z, rr, cc_img] = label_value

    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.SetOrigin(origin)
    mask_img.SetSpacing(spacing)
    mask_img.SetDirection(image.GetDirection())

    if action == 'replace_from_csv' and existing_mask_exists:
        try:
            _, mask_path, _ = _ensure_project_managed_mask(project, volume_id, existing_nii_path)
        except Exception:
            mask_path = _build_ct_init_mask_path(project.project_path, volume.get('name') or f'ct_volume_{volume_id}', volume_id)
    else:
        mask_path = _build_ct_init_mask_path(project.project_path, volume.get('name') or f'ct_volume_{volume_id}', volume_id)

    sitk.WriteImage(mask_img, mask_path)
    project.update_ct_volume_nii(volume_id, mask_path)

    action_messages = {
        'create_blank': '已创建空白掩码，可直接开始手工涂鸦',
        'create_from_csv': '已根据 CSV 标注生成初始掩码',
        'replace_from_csv': '已根据 CSV 标注重新生成并覆盖当前关联掩码',
    }
    return {
        'volume': project.get_ct_volume(volume_id) or volume,
        'nii_path': mask_path,
        'action': action,
        'message': action_messages[action],
    }


def _normalize_ct_classes(classes):
    """
    将项目类别转成 CT 页面可直接消费的结构。
    标签值从 1 开始，与 NIfTI 掩码标签值对应。
    """
    palette = [
        '#ff6432',
        '#32c864',
        '#3282ff',
        '#dc32dc',
        '#ffdc32',
        '#32dcdc',
        '#ff82c8',
        '#a0ff64',
    ]
    safe_classes = [str(c).strip() for c in (classes or []) if str(c).strip()]
    if not safe_classes:
        safe_classes = ['nodule']
    return [
        {
            'name': cls,
            'value': idx + 1,
            'color': palette[idx % len(palette)],
        }
        for idx, cls in enumerate(safe_classes)
    ]


def _get_ct_project(project_name: str):
    project_path = os.path.join(_projects_folder(), secure_filename(project_name))
    if not os.path.exists(project_path):
        return None, None
    return project_path, Project(project_name, '', '', project_path)


def _get_user_assigned_ct_volume_names(project_name: str, user_id: int) -> set[str]:
    assignments = get_user_assigned_images(project_name, user_id)
    return {
        str(item.get('image_name') or '').strip()
        for item in assignments
        if str(item.get('image_name') or '').strip()
    }


def _filter_ct_volumes_for_user(project_name: str, user, volumes: list[dict]) -> list[dict]:
    if getattr(user, 'role', '') != 'annotator':
        return list(volumes)
    assigned_names = _get_user_assigned_ct_volume_names(project_name, user.id)
    return [volume for volume in volumes if str(volume.get('name') or '') in assigned_names]


def _user_can_access_ct_volume(project_name: str, user, volume: dict | None) -> bool:
    if not volume:
        return False
    if getattr(user, 'role', '') != 'annotator':
        return True
    assigned_names = _get_user_assigned_ct_volume_names(project_name, user.id)
    return str(volume.get('name') or '') in assigned_names


def _medsam2_store_task(task_id: str, payload: dict):
    with _medsam2_task_lock:
        _medsam2_task_cache[task_id] = payload


def _medsam2_get_task(task_id: str) -> dict | None:
    with _medsam2_task_lock:
        task = _medsam2_task_cache.get(task_id)
        return dict(task) if task else None


def _get_volume_origin_and_spacing(volume_path: str):
    from medseg.utils.ct_utils import _load_sitk

    sitk = _load_sitk()
    reader = sitk.ImageFileReader()
    reader.SetFileName(volume_path)
    reader.ReadImageInformation()
    origin = reader.GetOrigin()
    spacing = reader.GetSpacing()
    return origin, spacing


def _world_to_voxel_point(coord_x: float, coord_y: float, coord_z: float, origin, spacing):
    voxel_x = int(round((float(coord_x) - float(origin[0])) / max(float(spacing[0]), 1e-6)))
    voxel_y = int(round((float(coord_y) - float(origin[1])) / max(float(spacing[1]), 1e-6)))
    voxel_z = int(round((float(coord_z) - float(origin[2])) / max(float(spacing[2]), 1e-6)))
    return voxel_x, voxel_y, voxel_z


def _build_default_medsam2_output_path(project: Project, volume: dict, volume_id: int) -> str:
    masks_dir = os.path.join(project.project_path, 'ct_volumes', 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    stem = os.path.splitext(os.path.splitext(os.path.basename(volume.get('name') or f'ct_volume_{volume_id}'))[0])[0]
    safe_stem = secure_filename(stem) or f'ct_volume_{volume_id}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(masks_dir, f'{safe_stem}_medsam2_{volume_id}_{timestamp}.nii.gz')


def _build_medsam2_prompts(project: Project, volume_id: int, volume_path: str, request_data: dict) -> tuple[list[list[float]], list[list[float]]]:
    points = []
    boxes = []

    raw_points = request_data.get('point_prompt') or request_data.get('points') or []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        try:
            points.append([float(item[0]), float(item[1]), float(item[2]), int(item[3])])
        except (TypeError, ValueError):
            continue

    raw_boxes = request_data.get('box_prompt') or request_data.get('boxes') or []
    for item in raw_boxes:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            boxes.append([float(v) for v in item[:6]])
        except (TypeError, ValueError):
            continue

    if points or boxes:
        return points, boxes

    annotations = project.get_ct_annotations_for_volume(volume_id)
    if not annotations:
        return points, boxes

    origin, spacing = _get_volume_origin_and_spacing(volume_path)
    for anno in annotations:
        try:
            voxel_x, voxel_y, voxel_z = _world_to_voxel_point(
                anno['coord_x'],
                anno['coord_y'],
                anno['coord_z'],
                origin,
                spacing,
            )
            points.append([voxel_x, voxel_y, voxel_z, 1])
        except Exception as exc:
            logger.warning("Failed to convert CT annotation to voxel prompt volume_id=%s anno_id=%s error=%s", volume_id, anno.get('anno_id'), exc)
    return points, boxes


def _has_mask_label_on_slice(nii_path: str, axis: str, slice_idx: int, label_value: int) -> bool:
    from medseg.utils.ct_utils import _get_cached_array

    arr = _get_cached_array(nii_path)
    axis = str(axis or 'z').lower()
    label_value = int(label_value)
    if axis == 'z':
        if slice_idx < 0 or slice_idx >= arr.shape[0]:
            return False
        return bool((arr[slice_idx, :, :] == label_value).any())
    if axis == 'y':
        if slice_idx < 0 or slice_idx >= arr.shape[1]:
            return False
        return bool((arr[:, slice_idx, :] == label_value).any())
    if axis == 'x':
        if slice_idx < 0 or slice_idx >= arr.shape[2]:
            return False
        return bool((arr[:, :, slice_idx] == label_value).any())
    return False


def _parse_volume_ids(raw_ids):
    volume_ids = []
    for value in raw_ids or []:
        try:
            volume_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return volume_ids


def _sanitize_export_base(name: str) -> str:
    stem = os.path.splitext(os.path.splitext(os.path.basename(name or 'ct_volume'))[0])[0]
    return secure_filename(stem) or 'ct_volume'


def _write_ct_mask_to_format(mask_path: str, target_format: str):
    import tempfile
    import SimpleITK as sitk

    image = sitk.ReadImage(mask_path)
    suffix_map = {
        'nifti': '.nii.gz',
        'nrrd': '.nrrd',
        'mha': '.mha',
        'mhd': '.mhd',
    }
    suffix = suffix_map.get(target_format, '.nii.gz')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    sitk.WriteImage(image, tmp.name)
    return tmp.name


def _build_ct_export_zip(project_name: str, project: Project, selected_volumes: list, export_format: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mem_zip = BytesIO()
    csv_rows = []
    with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        manifest_lines = [
            f'project: {project_name}',
            f'exported_at: {timestamp}',
            f'format: {export_format}',
            '',
            'files:'
        ]
        temp_paths = []
        try:
            for volume in selected_volumes:
                volume_id = volume['volume_id']
                volume_name = volume.get('name') or f'volume_{volume_id}'
                base_name = _sanitize_export_base(volume_name)
                managed_path = None

                if export_format in ('nifti', 'nrrd', 'mha', 'mhd'):
                    _, managed_path, _ = _ensure_project_managed_mask(project, volume_id, volume.get('nii_path'))
                    converted_path = _write_ct_mask_to_format(managed_path, export_format)
                    temp_paths.append(converted_path)
                    arc_dir = export_format
                    arc_name = os.path.join(arc_dir, f'{base_name}_{volume_id}{os.path.splitext(converted_path)[1]}')
                    if converted_path.endswith('.nii.gz'):
                        arc_name = os.path.join(arc_dir, f'{base_name}_{volume_id}.nii.gz')
                    elif converted_path.endswith('.mhd'):
                        arc_name = os.path.join(arc_dir, f'{base_name}_{volume_id}.mhd')
                    elif converted_path.endswith('.mha'):
                        arc_name = os.path.join(arc_dir, f'{base_name}_{volume_id}.mha')
                    elif converted_path.endswith('.nrrd'):
                        arc_name = os.path.join(arc_dir, f'{base_name}_{volume_id}.nrrd')
                    zf.write(converted_path, arcname=arc_name)
                    manifest_lines.append(f"- volume_id={volume_id} name={volume_name} mask={arc_name}")

                    if converted_path.endswith('.mhd'):
                        raw_path = os.path.splitext(converted_path)[0] + '.raw'
                        if os.path.isfile(raw_path):
                            temp_paths.append(raw_path)
                            zf.write(raw_path, arcname=os.path.join(arc_dir, f'{base_name}_{volume_id}.raw'))
                elif export_format != 'csv':
                    raise ValueError('不支持的导出格式')

                volume_rows = []
                for anno in project.get_ct_annotations_for_volume(volume_id):
                    csv_rows.append([
                        volume_id,
                        volume_name,
                        anno.get('label') or '',
                        anno.get('coord_x') or '',
                        anno.get('coord_y') or '',
                        anno.get('coord_z') or '',
                        anno.get('diameter_mm') or '',
                    ])
                    volume_rows.append(anno)

                if not volume_rows:
                    mask_path = managed_path or str(volume.get('nii_path') or '').strip()
                    if mask_path:
                        try:
                            for anno in _extract_ct_annotations_from_mask(mask_path, project.get_classes() or ['nodule']):
                                csv_rows.append([
                                    volume_id,
                                    volume_name,
                                    anno.get('label') or '',
                                    anno.get('coord_x') or '',
                                    anno.get('coord_y') or '',
                                    anno.get('coord_z') or '',
                                    anno.get('diameter_mm') or '',
                                ])
                        except Exception as export_error:
                            logger.warning(
                                "从掩码生成 CT 导出 CSV 失败 project=%s volume_id=%s mask=%s error=%s",
                                project_name,
                                volume_id,
                                mask_path,
                                export_error,
                            )
                manifest_lines.append(f"- volume_id={volume_id} name={volume_name}")

            csv_io = StringIO()
            writer = csv.writer(csv_io)
            writer.writerow(['volume_id', 'volume_name', 'label', 'coord_x', 'coord_y', 'coord_z', 'diameter_mm'])
            for row in csv_rows:
                writer.writerow(row)

            zf.writestr('annotations/ct_annotations.csv', csv_io.getvalue())
            zf.writestr('README.txt', '\n'.join(manifest_lines))
        finally:
            for path in temp_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    mem_zip.seek(0)
    return mem_zip, timestamp


def _extract_ct_annotations_from_mask(mask_path: str, project_classes: list[str] | None = None) -> list[dict]:
    if not mask_path or not os.path.isfile(mask_path):
        return []
    if not str(mask_path).lower().endswith(('.nii', '.nii.gz')):
        return []

    import numpy as np
    from scipy import ndimage
    from medseg.utils.ct_utils import _get_cached_array, get_nii_metadata

    arr = _get_cached_array(mask_path)
    mask = arr != 0
    if not np.any(mask):
        return []

    labeled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    objects = ndimage.find_objects(labeled)
    safe_classes = [str(item).strip() for item in (project_classes or []) if str(item).strip()]
    if not safe_classes:
        safe_classes = ['nodule']

    spacing_x = spacing_y = spacing_z = 1.0
    origin_x = origin_y = origin_z = 0.0
    try:
        meta = get_nii_metadata(mask_path)
        spacing = meta.get('spacing') or (1.0, 1.0, 1.0)
        origin = meta.get('origin') or (0.0, 0.0, 0.0)
        spacing_x = float(spacing[0])
        spacing_y = float(spacing[1])
        spacing_z = float(spacing[2])
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        origin_z = float(origin[2])
    except Exception:
        pass

    annotations = []
    for label_idx in range(1, count + 1):
        slc = objects[label_idx - 1]
        if not slc:
            continue
        component = labeled[slc] == label_idx
        raw_vals = arr[slc][component]
        raw_vals = raw_vals[raw_vals > 0]
        label_value = int(np.bincount(raw_vals.astype(int)).argmax()) if raw_vals.size else 1
        class_label = safe_classes[label_value - 1] if 1 <= label_value <= len(safe_classes) else f'class_{label_value}'

        z_len = max(1, slc[0].stop - slc[0].start) * spacing_z
        y_len = max(1, slc[1].stop - slc[1].start) * spacing_y
        x_len = max(1, slc[2].stop - slc[2].start) * spacing_x
        diameter = max(x_len, y_len, z_len)

        center_z_idx = (slc[0].start + slc[0].stop - 1) / 2.0
        center_y_idx = (slc[1].start + slc[1].stop - 1) / 2.0
        center_x_idx = (slc[2].start + slc[2].stop - 1) / 2.0

        annotations.append({
            'label': class_label,
            'label_value': label_value,
            'coord_x': round(float(origin_x + center_x_idx * spacing_x), 3),
            'coord_y': round(float(origin_y + center_y_idx * spacing_y), 3),
            'coord_z': round(float(origin_z + center_z_idx * spacing_z), 3),
            'diameter_mm': round(float(diameter), 3),
        })

    return annotations


def _get_selected_ct_volumes(project: Project, volume_ids: list[int]) -> list:
    volumes = project.get_ct_volumes()
    if not volume_ids:
        return volumes
    selected = set(volume_ids)
    return [volume for volume in volumes if int(volume['volume_id']) in selected]


@bp.route('/ct_volume_meta/<project_name>/<int:volume_id>')
@login_required
def ct_volume_meta(project_name, volume_id):
    """
    返回 CT 体数据的 MHD 头信息（JSON 格式），供前端 vtk.js 解析。
    包括：DimSize, ElementSpacing, ElementType, Origin
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'error': '您没有权限访问该 CT 任务'}), 403
    mhd_path = project.get_ct_volume_path(volume_id)
    if not mhd_path or not os.path.isfile(mhd_path):
        return jsonify({'error': 'CT 体数据文件不存在'}), 404

    try:
        meta = {}
        with open(mhd_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    meta[key.strip()] = val.strip()

        dim = list(map(int, meta.get('DimSize', '0 0 0').split()))
        spacing = list(map(float, meta.get('ElementSpacing', '1 1 1').split()))
        origin = list(map(float, meta.get('Offset', meta.get('Origin', '0 0 0')).split()))
        elem_type = meta.get('ElementType', 'MET_SHORT')

        # 构造 RAW 文件的下载 URL
        raw_url = f'/ct_volume_raw/{project_name}/{volume_id}'

        return jsonify({
            'success': True,
            'dim': dim,          # [X, Y, Z]
            'spacing': spacing,  # [sx, sy, sz] mm
            'origin': origin,    # [ox, oy, oz] mm
            'elem_type': elem_type,
            'raw_url': raw_url,
        })
    except Exception as e:
        logger.error(f"读取 MHD 元数据失败 {mhd_path}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/ct_volume_raw/<project_name>/<int:volume_id>')
@login_required
def ct_volume_raw(project_name, volume_id):
    """
    流式传输 CT 体数据的原始 RAW 二进制文件。
    前端用 fetch() + arrayBuffer() 一次性接收，在浏览器内存中完成所有切片渲染。
    支持 HTTP Range 请求，允许断点续传。
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'error': '您没有权限访问该 CT 任务'}), 403
    mhd_path = project.get_ct_volume_path(volume_id)
    if not mhd_path or not os.path.isfile(mhd_path):
        return jsonify({'error': 'CT 体数据文件不存在'}), 404

    # 从 MHD 头找到对应的 RAW 文件
    raw_path = None
    try:
        with open(mhd_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('ElementDataFile'):
                    raw_filename = line.split('=', 1)[1].strip()
                    raw_path = os.path.join(os.path.dirname(mhd_path), raw_filename)
                    break
    except Exception as e:
        logger.error(f"读取 RAW 路径失败: {e}")
        return jsonify({'error': '无法定位 RAW 文件'}), 500

    if not raw_path or not os.path.isfile(raw_path):
        return jsonify({'error': f'RAW 文件不存在: {raw_path}'}), 404

    # 设置 COOP/COEP 响应头，允许前端使用 SharedArrayBuffer 实现零拷贝内存读取
    response = send_file(
        raw_path,
        mimetype='application/octet-stream',
        conditional=True,  # 支持 HTTP Range 断点续传
    )
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.route('/ct/<project_name>')
@login_required
def ct_viewer(project_name):
    """CT 项目浏览页面"""
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return "项目不存在", 404

    project = Project(project_name, '', '', project_path)
    volumes_raw = _filter_ct_volumes_for_user(project_name, current_user, project.get_ct_volumes())
    classes = project.get_classes()

    # 为每个体数据计算预览图 URL
    volumes = []
    for v in volumes_raw:
        preview_url = None
        preview_path = v.get('preview_path')
        if preview_path and os.path.isfile(preview_path):
            try:
                rel = os.path.relpath(preview_path, PROJECTS_FOLDER).replace('\\', '/')
                preview_url = f'/projects/{rel}'
            except ValueError:
                preview_url = None
        v['preview_url'] = preview_url
        volumes.append(v)

    return render_template(
        'ct_viewer.html',
        project_name=project_name,
        volumes=volumes,
        classes=classes,
        ct_classes=_normalize_ct_classes(classes),
    )


@bp.route('/ct_slice/<project_name>/<int:volume_id>')
@login_required
def ct_slice(project_name, volume_id):
    """
    按需返回 CT 切片 PNG 图像。
    查询参数：
      axis  : z(轴状面，默认) | y(冠状面) | x(矢状面)
      index : 切片索引（0-based）
      ww    : 窗宽 (默认 1500，肺窗)
      wc    : 窗位 (默认 -600，肺窗)
    """
    axis = request.args.get('axis', 'z').lower()
    try:
        index = int(request.args.get('index', 0))
        ww = float(request.args.get('ww', 1500))
        wc = float(request.args.get('wc', -600))
    except (ValueError, TypeError):
        return jsonify({'error': '参数格式错误'}), 400

    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'error': '您没有权限访问该 CT 任务'}), 403
    mhd_path = project.get_ct_volume_path(volume_id)
    if not mhd_path or not os.path.isfile(mhd_path):
        return jsonify({'error': 'CT 体数据文件不存在'}), 404

    # 从数据库取体素间距，用于比例校正
    import sqlite3
    spacing_x = spacing_y = spacing_z = 1.0
    try:
        db_path = os.path.join(project_path, 'config.db')
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                'SELECT spacing_x, spacing_y, spacing_z FROM CTVolumes WHERE volume_id = ?',
                (volume_id,)
            ).fetchone()
            if row:
                spacing_x, spacing_y, spacing_z = float(row[0]), float(row[1]), float(row[2])
    except Exception as e:
        logger.warning(f"读取 spacing 失败 volume_id={volume_id}: {e}")

    try:
        from medseg.utils.ct_utils import get_slice_jpeg
        jpeg_bytes = get_slice_jpeg(
            mhd_path, axis, index,
            ww=ww, wc=wc,
            spacing_x=spacing_x, spacing_y=spacing_y, spacing_z=spacing_z
        )
        return Response(
            jpeg_bytes,
            mimetype='image/jpeg',
            headers={
                'Cache-Control': 'no-cache',
                'X-Slice-Axis': axis,
                'X-Slice-Index': str(index),
            }
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"切片渲染失败 project={project_name} volume={volume_id} axis={axis} idx={index}: {e}")
        return jsonify({'error': '切片渲染失败'}), 500


@bp.route('/ct_volumes/<project_name>', methods=['GET'])
@login_required
def get_ct_volumes(project_name):
    """返回项目内所有 CT 体数据列表（JSON）"""
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volumes = _filter_ct_volumes_for_user(project_name, current_user, project.get_ct_volumes())

    # 将预览图路径转换为 URL
    result = []
    for v in volumes:
        preview_url = None
        preview_path = v.get('preview_path')
        if preview_path and os.path.isfile(preview_path):
            try:
                rel = os.path.relpath(preview_path, PROJECTS_FOLDER).replace('\\', '/')
                preview_url = f'/projects/{rel}'
            except ValueError:
                preview_url = None
        result.append({
            'volume_id': v['volume_id'],
            'name': v['name'],
            'shape_x': v['shape_x'],
            'shape_y': v['shape_y'],
            'shape_z': v['shape_z'],
            'spacing_x': v['spacing_x'],
            'spacing_y': v['spacing_y'],
            'spacing_z': v['spacing_z'],
            'preview_url': preview_url,
            'added_at': v['added_at'],
            'nii_path': v.get('nii_path') or '',
            'has_mask': bool(v.get('nii_path') and os.path.isfile(v['nii_path'])),
        })

    return jsonify({'success': True, 'volumes': result})


@bp.route('/download_ct_volumes/<project_name>', methods=['POST'])
@login_required
def download_ct_volumes(project_name):
    project_path, project = _get_ct_project(project_name)
    if not project:
        return jsonify({'success': False, 'error': '项目不存在'}), 404

    data = request.get_json(silent=True) or {}
    volume_ids = _parse_volume_ids(data.get('volume_ids', []))
    selected_volumes = _filter_ct_volumes_for_user(
        project_name,
        current_user,
        _get_selected_ct_volumes(project, volume_ids),
    )
    if not selected_volumes:
        return jsonify({'success': False, 'error': '没有可下载的 CT 文件'}), 400

    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for volume in selected_volumes:
            volume_id = volume['volume_id']
            volume_name = volume.get('name') or f'volume_{volume_id}'
            base_name = _sanitize_export_base(volume_name)
            absolute_path = volume.get('absolute_path') or ''
            if os.path.isfile(absolute_path):
                ext = '.nii.gz' if absolute_path.lower().endswith('.nii.gz') else os.path.splitext(absolute_path)[1]
                zf.write(absolute_path, arcname=os.path.join('images', f'{base_name}_{volume_id}{ext}'))

                if absolute_path.lower().endswith('.mhd'):
                    try:
                        with open(absolute_path, 'r', encoding='utf-8', errors='ignore') as fh:
                            for line in fh:
                                if line.strip().startswith('ElementDataFile'):
                                    raw_name = line.split('=', 1)[1].strip()
                                    raw_path = os.path.join(os.path.dirname(absolute_path), raw_name)
                                    if os.path.isfile(raw_path):
                                        zf.write(raw_path, arcname=os.path.join('images', os.path.basename(raw_path)))
                                    break
                    except Exception as error:
                        logger.warning(f"打包 CT 原始 RAW 失败 {absolute_path}: {error}")

            preview_path = volume.get('preview_path') or ''
            if os.path.isfile(preview_path):
                zf.write(preview_path, arcname=os.path.join('previews', os.path.basename(preview_path)))

    mem_zip.seek(0)
    return send_file(
        mem_zip,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{project_name}_ct_volumes.zip'
    )


@bp.route('/delete_ct_volumes/<project_name>', methods=['POST'])
@login_required
def delete_ct_volumes(project_name):
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '只有管理员可以删除 CT 数据'}), 403
    _, project = _get_ct_project(project_name)
    if not project:
        return jsonify({'success': False, 'error': '项目不存在'}), 404

    data = request.get_json(silent=True) or {}
    volume_ids = _parse_volume_ids(data.get('volume_ids', []))
    if not volume_ids:
        return jsonify({'success': False, 'error': '未选择需要删除的 CT 文件'}), 400

    deleted = project.delete_ct_volumes(volume_ids)
    return jsonify({'success': True, 'deleted': deleted})


@bp.route('/save_ct_annotations/<project_name>/<int:volume_id>', methods=['POST'])
@login_required
def save_ct_annotations(project_name, volume_id):
    """
    显式保存当前 CT 掩码文件到项目内，并将数据库中的 nii_path 固定为项目管理路径。
    现有编辑接口本身已经按笔实时写盘，这里提供与二维标注一致的“保存确认”入口。
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    access_volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, access_volume):
        return jsonify({'success': False, 'error': '您没有权限保存该 CT 任务'}), 403
    data = request.get_json(silent=True) or {}
    nii_path = str(data.get('nii_path') or '').strip()

    try:
        volume, managed_path, copied = _ensure_project_managed_mask(project, volume_id, nii_path)
        try:
            project.mark_ct_volume_reviewed(volume_id, getattr(current_user, 'id', None))
        except Exception as review_error:
            logger.warning("Failed to mark CT volume reviewed project=%s volume=%s error=%s", project_name, volume_id, review_error)
        try:
            update_image_status(project_name, str(volume.get('name') or ''), 'completed')
        except Exception as status_error:
            logger.warning("Failed to update CT assignment status project=%s volume=%s error=%s", project_name, volume_id, status_error)
        return jsonify({
            'success': True,
            'volume_id': volume_id,
            'nii_path': managed_path,
            'copied_to_project': copied,
            'message': 'CT 标注已保存'
        })
    except FileNotFoundError as error:
        return jsonify({'success': False, 'error': str(error)}), 404
    except Exception as error:
        logger.error(f"保存 CT 标注失败 project={project_name} volume={volume_id}: {error}")
        return jsonify({'success': False, 'error': str(error)}), 500


@bp.route('/export_ct_annotations/<project_name>', methods=['GET'])
@login_required
def export_ct_annotations(project_name):
    """
    导出 CT 项目中的标注文件。
    当前导出内容以每个体数据对应的 NIfTI 掩码文件为主，打包为 zip 下载。
    """
    project_path, project = _get_ct_project(project_name)
    if not project:
        return jsonify({'success': False, 'error': '项目不存在'}), 404
    export_format = (request.args.get('format', 'nifti') or 'nifti').strip().lower()
    volume_ids = _parse_volume_ids(request.args.getlist('volume_ids'))
    selected_volumes = _filter_ct_volumes_for_user(
        project_name,
        current_user,
        _get_selected_ct_volumes(project, volume_ids),
    )
    if not selected_volumes:
        return jsonify({'success': False, 'error': '没有可导出的 CT 文件'}), 400

    try:
        mem_zip, timestamp = _build_ct_export_zip(project_name, project, selected_volumes, export_format)
    except FileNotFoundError as error:
        return jsonify({'success': False, 'error': str(error)}), 404
    except ValueError as error:
        return jsonify({'success': False, 'error': str(error)}), 400
    except Exception as error:
        logger.error(f"导出 CT 标注失败 project={project_name}: {error}")
        return jsonify({'success': False, 'error': str(error)}), 500

    return send_file(
        mem_zip,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{project_name}_ct_annotations_{export_format}_{timestamp}.zip'
    )


# ==============================================================
# NIfTI 掩码可视化与转换 API
# ==============================================================

@bp.route('/init_nii_mask/<project_name>/<int:volume_id>', methods=['POST'])
@login_required
def init_nii_mask(project_name, volume_id):
    """
    初始化 CT 掩码文件。
    支持三种动作：
    - create_blank: 创建空白掩码
    - create_from_csv: 根据 CSV/数据库结节点生成初始掩码
    - replace_from_csv: 强制根据 CSV/数据库结节点重建并覆盖当前关联掩码
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'error': '您没有权限访问该 CT 任务'}), 403
    data = request.get_json(silent=True) or {}
    action = str(data.get('action') or 'create_blank').strip().lower()
    try:
        result = initialize_ct_volume_mask(project, volume_id, action=action)
        return jsonify({
            'success': True,
            'nii_path': result['nii_path'],
            'message': result['message'],
            'action': result['action'],
        })
    except FileExistsError as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to initialize blank NIfTI mask: {e}")
        return jsonify({'error': str(e)}), 500

# ==============================================================

@bp.route('/nii_mask_slice')
@login_required
def nii_mask_slice():
    """
    返回 NIfTI 分割掩码切片 RGBA PNG。
    查询参数：
      path  : NIfTI 文件的绝对路径（服务端本地文件系统）
      axis  : z(轴状，默认) | y(冠状) | x(矢状)
      index : 切片索引 (0-based)
      sx/sy/sz : 体素间距 mm（用于比例校正，默认 1.0）
      alpha : 掩码不透明度 0~255（默认 180）
    返回：RGBA PNG 图像，背景全透明
    """
    nii_path = request.args.get('path', '').strip()
    axis = request.args.get('axis', 'z').lower()
    try:
        index = int(request.args.get('index', 0))
        sx = float(request.args.get('sx', 1.0))
        sy = float(request.args.get('sy', 1.0))
        sz = float(request.args.get('sz', 1.0))
        alpha = int(request.args.get('alpha', 180))
    except (ValueError, TypeError):
        return jsonify({'error': '参数格式错误'}), 400

    if not nii_path:
        return jsonify({'error': '缺少 path 参数'}), 400

    if not os.path.isfile(nii_path):
        return jsonify({'error': f'NIfTI 文件不存在: {nii_path}'}), 404

    p_lower = nii_path.lower()
    if not (p_lower.endswith('.nii') or p_lower.endswith('.nii.gz')):
        return jsonify({'error': '仅支持 .nii 或 .nii.gz 格式'}), 400

    try:
        color_str = request.args.get('color', '').strip()
        color_rgb = None
        if color_str:
            try:
                parts = [int(x) for x in color_str.split(',')]
                if len(parts) == 3:
                    color_rgb = tuple(parts)
            except Exception:
                pass

        from medseg.utils.ct_utils import get_nii_mask_slice_png
        png_bytes = get_nii_mask_slice_png(
            nii_path, axis, index,
            spacing_x=sx, spacing_y=sy, spacing_z=sz,
            alpha=alpha, color_rgb=color_rgb
        )
        return Response(
            png_bytes,
            mimetype='image/png',
            headers={
                'Cache-Control': 'no-cache',
                'X-Slice-Axis': axis,
                'X-Slice-Index': str(index),
            }
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"NIfTI 切片渲染失败 path={nii_path} axis={axis} idx={index}: {e}")
        return jsonify({'error': 'NIfTI 切片渲染失败'}), 500


@bp.route('/nii_meta')
@login_required
def nii_meta():
    """
    返回 NIfTI 文件的元数据（shape、spacing 等）以及标签列表。
    查询参数：
      path : NIfTI 文件的绝对路径
    """
    nii_path = request.args.get('path', '').strip()
    if not nii_path:
        return jsonify({'error': '缺少 path 参数'}), 400
    if not os.path.isfile(nii_path):
        return jsonify({'error': f'文件不存在: {nii_path}'}), 404

    p_lower = nii_path.lower()
    if not (p_lower.endswith('.nii') or p_lower.endswith('.nii.gz')):
        return jsonify({'error': '仅支持 .nii 或 .nii.gz 格式'}), 400

    try:
        from medseg.utils.ct_utils import get_nii_metadata, list_nii_labels
        meta = get_nii_metadata(nii_path)
        labels = list_nii_labels(nii_path)
        return jsonify({'success': True, 'meta': meta, 'labels': labels})
    except Exception as e:
        logger.error(f"NIfTI 元数据解析失败 {nii_path}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/nii_mask_slices')
@login_required
def nii_mask_slices():
    """
    返回 NIfTI 掩码中各轴上含有非零标注的切片索引列表。
    用于前端在进度条上渲染掩码标记点，方便用户快速定位标注切片。
    查询参数：path : NIfTI 文件的绝对路径
    返回：{z:[int...], y:[int...], x:[int...]}
    """
    nii_path = request.args.get('path', '').strip()
    if not nii_path:
        return jsonify({'error': '缺少 path 参数'}), 400
    if not os.path.isfile(nii_path):
        return jsonify({'error': f'文件不存在: {nii_path}'}), 404
    p_lower = nii_path.lower()
    if not (p_lower.endswith('.nii') or p_lower.endswith('.nii.gz')):
        return jsonify({'error': '仅支持 .nii 或 .nii.gz 格式'}), 400
    try:
        import numpy as np
        from medseg.utils.ct_utils import _get_cached_array
        arr = _get_cached_array(nii_path)  # shape: (z, y, x)
        z_has = np.any(arr != 0, axis=(1, 2))
        y_has = np.any(arr != 0, axis=(0, 2))
        x_has = np.any(arr != 0, axis=(0, 1))

        z_marks = {}
        y_marks = {}
        x_marks = {}

        for i in np.where(z_has)[0]:
            labels = [int(v) for v in np.unique(arr[int(i), :, :]) if v != 0]
            z_marks[int(i)] = labels
        for i in np.where(y_has)[0]:
            labels = [int(v) for v in np.unique(arr[:, int(i), :]) if v != 0]
            y_marks[int(i)] = labels
        for i in np.where(x_has)[0]:
            labels = [int(v) for v in np.unique(arr[:, :, int(i)]) if v != 0]
            x_marks[int(i)] = labels

        return jsonify({
            'success': True,
            'z': [int(i) for i in np.where(z_has)[0]],
            'y': [int(i) for i in np.where(y_has)[0]],
            'x': [int(i) for i in np.where(x_has)[0]],
            'z_marks': z_marks,
            'y_marks': y_marks,
            'x_marks': x_marks,
        })
    except Exception as e:
        logger.error(f"计算掩码切片分布失败 {nii_path}: {e}")
        return jsonify({'error': str(e)}), 500


# undo stacks: {nii_path: [(axis, slice_idx, slice_data_copy), ...]}
_undo_stacks = {}
_MAX_UNDO = 12
_overlap_backups = {}


def _overlap_key(nii_path, axis, slice_idx):
    return (os.path.abspath(nii_path), str(axis), int(slice_idx))


def _snapshot_overlap_backups(nii_path, axis, slice_idx):
    key = _overlap_key(nii_path, axis, slice_idx)
    return [
        {
            'top_label': int(entry['top_label']),
            'top_mask': entry['top_mask'].copy(),
            'under_values': entry['under_values'].copy(),
        }
        for entry in _overlap_backups.get(key, [])
    ]


def _restore_overlap_backup_snapshot(nii_path, axis, slice_idx, snapshot):
    key = _overlap_key(nii_path, axis, slice_idx)
    if snapshot:
        _overlap_backups[key] = [
            {
                'top_label': int(entry['top_label']),
                'top_mask': entry['top_mask'].copy(),
                'under_values': entry['under_values'].copy(),
            }
            for entry in snapshot
        ]
    else:
        _overlap_backups.pop(key, None)


def _clear_overlap_backups(nii_path, axis=None, slice_idx=None):
    if axis is not None and slice_idx is not None:
        _overlap_backups.pop(_overlap_key(nii_path, axis, slice_idx), None)
        return
    path = os.path.abspath(nii_path)
    for key in list(_overlap_backups.keys()):
        if key[0] == path:
            _overlap_backups.pop(key, None)


def _push_slice_undo(nii_path, axis, slice_idx, slice_data):
    if nii_path not in _undo_stacks:
        _undo_stacks[nii_path] = []
    backup_snapshot = _snapshot_overlap_backups(nii_path, axis, slice_idx)
    _undo_stacks[nii_path].append((axis, slice_idx, slice_data.copy(), backup_snapshot))
    if len(_undo_stacks[nii_path]) > _MAX_UNDO:
        old_entry = _undo_stacks[nii_path].pop(0)
        if isinstance(old_entry, dict):
            backup_path = old_entry.get('backup_path')
            if backup_path and os.path.isfile(backup_path):
                try:
                    os.remove(backup_path)
                except OSError:
                    pass


def _push_volume_restore_undo(nii_path: str, backup_path: str):
    if nii_path not in _undo_stacks:
        _undo_stacks[nii_path] = []
    _undo_stacks[nii_path].append({
        'kind': 'volume_restore',
        'backup_path': os.path.abspath(backup_path),
    })
    if len(_undo_stacks[nii_path]) > _MAX_UNDO:
        old_entry = _undo_stacks[nii_path].pop(0)
        if isinstance(old_entry, dict):
            old_backup = old_entry.get('backup_path')
            if old_backup and os.path.isfile(old_backup):
                try:
                    os.remove(old_backup)
                except OSError:
                    pass


@bp.route('/nii_mask_edit', methods=['POST'])
@login_required
def nii_mask_edit():
    """
    画笔/橡皮擦编辑 NIfTI 掩码。
    请求体 JSON:
      nii_path  : NIfTI 文件绝对路径
      axis      : z/y/x (当前视图轴)
      slice_idx : 当前切片索引
      points    : [{vz,vy,vx}, ...] 笔触体素坐标列表
      radius    : 画笔半径（体素），默认 3
      label     : 1=涂抹(添加掩码)  0=橡皮擦(删除掩码)
    """
    import numpy as np
    data = request.get_json(silent=True) or {}
    nii_path  = data.get('nii_path', '').strip()
    axis      = data.get('axis', 'z')
    slice_idx = int(data.get('slice_idx', 0))
    points    = data.get('points', [])
    radius    = max(0, min(int(data.get('radius', 3)), 50))
    label     = float(data.get('label', 1))
    if not nii_path or not os.path.isfile(nii_path):
        return jsonify({'error': 'NIfTI 文件不存在'}), 404

    p_l = nii_path.lower()
    if not (p_l.endswith('.nii') or p_l.endswith('.nii.gz')):
        return jsonify({'error': '仅支持 .nii/.nii.gz'}), 400

    try:
        from medseg.utils.ct_utils import _get_cached_array, _load_sitk, clear_volume_cache
        arr = _get_cached_array(nii_path)   # (Z, Y, X) float32
        Z, Y, X = arr.shape

        # — push undo state (store only the affected 2D slice) —
        if axis == 'z':
            undo_data = arr[slice_idx, :, :].copy()
        elif axis == 'y':
            undo_data = arr[:, slice_idx, :].copy()
        else:
            undo_data = arr[:, :, slice_idx].copy()
        _push_slice_undo(nii_path, axis, slice_idx, undo_data)
        _clear_overlap_backups(nii_path, axis, slice_idx)

        # — apply brush strokes —
        r2 = radius * radius
        for pt in points:
            vz = max(0, min(int(pt.get('vz', 0)), Z-1))
            vy = max(0, min(int(pt.get('vy', 0)), Y-1))
            vx = max(0, min(int(pt.get('vx', 0)), X-1))
            if axis == 'z':
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        if dy*dy+dx*dx <= r2:
                            ny,nx = vy+dy, vx+dx
                            if 0<=ny<Y and 0<=nx<X:
                                arr[vz, ny, nx] = label
            elif axis == 'y':
                for dz in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        if dz*dz+dx*dx <= r2:
                            nz,nx = vz+dz, vx+dx
                            if 0<=nz<Z and 0<=nx<X:
                                arr[nz, vy, nx] = label
            else:
                for dz in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dz*dz+dy*dy <= r2:
                            nz,ny = vz+dz, vy+dy
                            if 0<=nz<Z and 0<=ny<Y:
                                arr[nz, ny, vx] = label

        # — save back (SimpleITK keeps same axis order as read) —
        sitk_lib = _load_sitk()
        ref = sitk_lib.ReadImage(nii_path)
        new_img = sitk_lib.GetImageFromArray(arr.astype(np.int16))
        new_img.CopyInformation(ref)
        sitk_lib.WriteImage(new_img, nii_path)
        clear_volume_cache(nii_path)
        return jsonify({'success': True, 'target_layers': ['gt']})
    except Exception as e:
        logger.error(f"NIfTI 编辑失败: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/nii_mask_undo', methods=['POST'])
@login_required
def nii_mask_undo():
    """撤销最近一次画笔操作"""
    import numpy as np

    data = request.get_json(silent=True) or {}
    nii_path = data.get('nii_path', '').strip()
    undo_path = nii_path if (nii_path in _undo_stacks and _undo_stacks[nii_path]) else None
    if not nii_path or not undo_path:
        return jsonify({'error': '无可撤销操作'}), 400
    try:
        undo_entry = _undo_stacks[undo_path].pop()
        from medseg.utils.ct_utils import _get_cached_array, _load_sitk, clear_volume_cache

        if isinstance(undo_entry, dict) and undo_entry.get('kind') == 'volume_restore':
            backup_path = str(undo_entry.get('backup_path') or '').strip()
            if not backup_path or not os.path.isfile(backup_path):
                return jsonify({'error': '撤销快照不存在，无法恢复'}), 400
            shutil.copy2(backup_path, undo_path)
            clear_volume_cache(undo_path)
            _clear_overlap_backups(undo_path)
            try:
                os.remove(backup_path)
            except OSError:
                pass
            return jsonify({'success': True, 'target_layer': 'gt', 'undo_kind': 'volume_restore'})

        if len(undo_entry) == 4:
            axis, slice_idx, undo_data, backup_snapshot = undo_entry
        else:
            axis, slice_idx, undo_data = undo_entry
            backup_snapshot = None
        arr = _get_cached_array(undo_path)
        if axis == 'z':
            arr[slice_idx, :, :] = undo_data
        elif axis == 'y':
            arr[:, slice_idx, :] = undo_data
        else:
            arr[:, :, slice_idx] = undo_data
        _restore_overlap_backup_snapshot(undo_path, axis, slice_idx, backup_snapshot)

        sitk_lib = _load_sitk()
        ref = sitk_lib.ReadImage(undo_path)
        new_img = sitk_lib.GetImageFromArray(arr.astype(np.int16))
        new_img.CopyInformation(ref)
        sitk_lib.WriteImage(new_img, undo_path)
        clear_volume_cache(undo_path)
        return jsonify({'success': True, 'target_layer': 'gt'})
    except Exception as e:
        logger.error(f"撤销 CT 标注失败: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/nii_mask_polygon', methods=['POST'])
@login_required
def nii_mask_polygon():
    """
    多边形填充编辑 NIfTI 掩码 (点连线方式)。
    请求体 JSON:
      nii_path  : NIfTI 文件绝对路径
      axis      : z/y/x
      slice_idx : 当前切片索引
      polygon   : [{cx,cy}, ...] 多边形顶点（画布像素坐标）
      canvas_w  : 渲染画布宽度（像素）
      canvas_h  : 渲染画布高度（像素）
      label     : 1=添加掩码  0=删除掩码（橡皮擦多边形）
    """
    import numpy as np
    data = request.get_json(silent=True) or {}
    nii_path  = data.get('nii_path', '').strip()
    axis      = data.get('axis', 'z')
    slice_idx = int(data.get('slice_idx', 0))
    pts       = data.get('polygon', [])          # [{cx,cy}, ...]
    canvas_w  = max(1.0, float(data.get('canvas_w', 512)))
    canvas_h  = max(1.0, float(data.get('canvas_h', 512)))
    label     = float(data.get('label', 1))
    if not nii_path or not os.path.isfile(nii_path):
        return jsonify({'error': 'NIfTI 文件不存在'}), 404

    if len(pts) < 3:
        return jsonify({'error': '至少需要 3 个顶点'}), 400

    try:
        from medseg.utils.ct_utils import _get_cached_array, _load_sitk
        from skimage.draw import polygon as ski_polygon

        arr = _get_cached_array(nii_path)   # (Z, Y, X) float32
        Z, Y, X = arr.shape

        # push undo state
        if axis == 'z':
            _push_slice_undo(nii_path, 'z', slice_idx, arr[slice_idx, :, :])
        elif axis == 'y':
            _push_slice_undo(nii_path, 'y', slice_idx, arr[:, slice_idx, :])
        else:
            _push_slice_undo(nii_path, 'x', slice_idx, arr[:, :, slice_idx])
        _clear_overlap_backups(nii_path, axis, slice_idx)

        # map canvas → voxel and rasterize polygon
        if axis == 'z':    # slice (Y,X), cx→X, cy→Y
            rows = np.array([p['cy'] / canvas_h * Y for p in pts])
            cols = np.array([p['cx'] / canvas_w * X for p in pts])
            rr, cc = ski_polygon(rows, cols, shape=(Y, X))
            arr[slice_idx, rr, cc] = label
        elif axis == 'y':  # slice (Z,X), cx→X, cy→reversed Z
            rows = np.array([(1.0 - (p['cy'] / canvas_h)) * Z for p in pts])
            cols = np.array([p['cx'] / canvas_w * X for p in pts])
            rr, cc = ski_polygon(rows, cols, shape=(Z, X))
            arr[rr, slice_idx, cc] = label
        else:              # slice (Z,Y), cx→Y, cy→reversed Z
            rows = np.array([(1.0 - (p['cy'] / canvas_h)) * Z for p in pts])
            cols = np.array([p['cx'] / canvas_w * Y for p in pts])
            rr, cc = ski_polygon(rows, cols, shape=(Z, Y))
            arr[rr, cc, slice_idx] = label

        sitk_lib = _load_sitk()
        ref = sitk_lib.ReadImage(nii_path)
        new_img = sitk_lib.GetImageFromArray(arr.astype(np.int16))
        new_img.CopyInformation(ref)
        sitk_lib.WriteImage(new_img, nii_path)
        from medseg.utils.ct_utils import clear_volume_cache
        clear_volume_cache(nii_path)

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"多边形掩码编辑失败: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/nii_mask_erase_blob', methods=['POST'])
@login_required
def nii_mask_erase_blob():
    """
    根据体素坐标 (vx, vy, vz) 进行 2D 连通域 (Blob) 的完美像素清除。
    """
    import numpy as np
    data = request.get_json(silent=True) or {}
    nii_path  = data.get('nii_path', '').strip()
    axis      = data.get('axis', 'z')
    slice_idx = int(data.get('slice_idx', 0))
    vx        = int(data.get('vx', 0))
    vy        = int(data.get('vy', 0))
    vz        = int(data.get('vz', 0))
    expected_label = int(float(data.get('label', 0) or 0))

    if not nii_path or not os.path.isfile(nii_path):
        return jsonify({'error': 'NIfTI 文件不存在'}), 404

    try:
        from medseg.utils.ct_utils import _get_cached_array, _load_sitk
        
        arr = _get_cached_array(nii_path)
        Z, Y, X = arr.shape
        slice_idx = max(0, min(slice_idx, {'z': Z - 1, 'y': Y - 1, 'x': X - 1}[axis]))
        vx = max(0, min(vx, X-1))
        vy = max(0, min(vy, Y-1))
        vz = max(0, min(vz, Z-1))
        target_path = nii_path
        if axis == 'z':
            label_value = int(arr[slice_idx, vy, vx])
        elif axis == 'y':
            label_value = int(arr[vz, slice_idx, vx])
        else:
            label_value = int(arr[vz, vy, slice_idx])

        if expected_label > 0:
            label_value = expected_label
        if label_value == 0:
            return jsonify({'error': '未找到可删除的标注区域'}), 400

        seed_rc = find_seed_in_slice(arr, axis, slice_idx, vx, vy, vz, label_value)
        if seed_rc is None:
            return jsonify({'error': f'未找到类别值为 {label_value} 的目标区域'}), 400

        # Push to undo stack
        if axis == 'z':
            _push_slice_undo(target_path, 'z', slice_idx, arr[slice_idx, :, :])
        elif axis == 'y':
            _push_slice_undo(target_path, 'y', slice_idx, arr[:, slice_idx, :])
        else:
            _push_slice_undo(target_path, 'x', slice_idx, arr[:, :, slice_idx])
        _clear_overlap_backups(target_path, axis, slice_idx)

        # 2D flood fill on the specified slice
        if axis == 'z':
            slice_2d = arr[slice_idx, :, :]
            seed_r, seed_c = seed_rc
            flood_fill_2d(slice_2d, seed_r, seed_c, 0, target_val=label_value)
            arr[slice_idx, :, :] = slice_2d
        elif axis == 'y':
            slice_2d = arr[:, slice_idx, :]
            seed_r, seed_c = seed_rc
            flood_fill_2d(slice_2d, seed_r, seed_c, 0, target_val=label_value)
            arr[:, slice_idx, :] = slice_2d
        else:
            slice_2d = arr[:, :, slice_idx]
            seed_r, seed_c = seed_rc
            flood_fill_2d(slice_2d, seed_r, seed_c, 0, target_val=label_value)
            arr[:, :, slice_idx] = slice_2d

        sitk_lib = _load_sitk()
        ref = sitk_lib.ReadImage(target_path)
        new_img = sitk_lib.GetImageFromArray(arr.astype(np.int16))
        new_img.CopyInformation(ref)
        sitk_lib.WriteImage(new_img, target_path)
        from medseg.utils.ct_utils import clear_volume_cache
        clear_volume_cache(target_path)

        return jsonify({'success': True, 'target_layer': 'gt', 'label': label_value})
    except Exception as e:
        logger.error(f"Erase blob failed: {e}")
        return jsonify({'error': str(e)}), 500

def find_seed_in_slice(arr, axis, slice_idx, vx, vy, vz, label_value, max_radius=48):
    """
    在当前 2D 切片内，为指定类别值找到一个可用于 flood fill 的种子点。
    如果用户传来的点已不在掩码内部，会在附近搜索同类标签，避免移动/缩放后旧区域删不掉。
    返回 (row, col) 或 None。
    """
    import numpy as np

    if axis == 'z':
        slice_2d = arr[slice_idx, :, :]
        start_r, start_c = vy, vx
    elif axis == 'y':
        slice_2d = arr[:, slice_idx, :]
        start_r, start_c = vz, vx
    else:
        slice_2d = arr[:, :, slice_idx]
        start_r, start_c = vz, vy

    rows, cols = slice_2d.shape
    start_r = max(0, min(start_r, rows - 1))
    start_c = max(0, min(start_c, cols - 1))

    if int(slice_2d[start_r, start_c]) == int(label_value):
        return start_r, start_c

    mask = (slice_2d == label_value)
    if not np.any(mask):
        return None

    rr, cc = np.where(mask)
    dist2 = (rr - start_r) ** 2 + (cc - start_c) ** 2
    best_idx = int(np.argmin(dist2))
    if dist2[best_idx] > max_radius * max_radius:
        return None
    return int(rr[best_idx]), int(cc[best_idx])

def flood_fill_2d(arr_2d, start_r, start_c, new_val=0, target_val=None):
    val = arr_2d[start_r, start_c] if target_val is None else target_val
    if val == new_val or arr_2d[start_r, start_c] != val:
        return
    R, C = arr_2d.shape
    queue = [(start_r, start_c)]
    visited = {(start_r, start_c)}
    arr_2d[start_r, start_c] = new_val
    
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                if (nr, nc) not in visited and arr_2d[nr, nc] == val:
                    visited.add((nr, nc))
                    arr_2d[nr, nc] = new_val
                    queue.append((nr, nc))


def connected_component_mask_2d(arr_2d, start_r, start_c, label_value):
    """
    返回当前 2D 切片中以种子点所在区域为准的同类连通域布尔 mask。
    选中标注的缩放/移动/旋转必须基于真实 NIfTI 像素，而不是前端 overlay 轮廓。
    """
    import numpy as np

    target = int(label_value)
    if int(arr_2d[start_r, start_c]) != target:
        return None
    rows, cols = arr_2d.shape
    component = np.zeros((rows, cols), dtype=bool)
    queue = [(int(start_r), int(start_c))]
    component[start_r, start_c] = True
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not component[nr, nc]:
                if int(arr_2d[nr, nc]) == target:
                    component[nr, nc] = True
                    queue.append((nr, nc))
    return component


def transform_component_mask_2d(component_mask, center_c, center_r, translate_c, translate_r, scale, angle_rad):
    """
    按前端画布交互参数变换真实连通域。返回目标位置的布尔 mask。
    """
    import math
    import numpy as np

    rows, cols = component_mask.shape
    scale = max(0.02, min(50.0, float(scale)))
    angle = float(angle_rad)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rr, cc = np.where(component_mask)
    if rr.size == 0:
        return np.zeros_like(component_mask, dtype=bool)

    src_c = cc.astype(np.float64) - float(center_c)
    src_r = rr.astype(np.float64) - float(center_r)
    dst_c = float(center_c) + scale * (src_c * cos_a - src_r * sin_a) + float(translate_c)
    dst_r = float(center_r) + scale * (src_c * sin_a + src_r * cos_a) + float(translate_r)

    min_c = max(0, int(math.floor(np.min(dst_c))) - 2)
    max_c = min(cols - 1, int(math.ceil(np.max(dst_c))) + 2)
    min_r = max(0, int(math.floor(np.min(dst_r))) - 2)
    max_r = min(rows - 1, int(math.ceil(np.max(dst_r))) + 2)
    if min_c > max_c or min_r > max_r:
        return np.zeros_like(component_mask, dtype=bool)

    grid_r, grid_c = np.mgrid[min_r:max_r + 1, min_c:max_c + 1]
    rel_c = grid_c.astype(np.float64) - float(translate_c) - float(center_c)
    rel_r = grid_r.astype(np.float64) - float(translate_r) - float(center_r)

    inv_c = (rel_c * cos_a + rel_r * sin_a) / scale + float(center_c)
    inv_r = (-rel_c * sin_a + rel_r * cos_a) / scale + float(center_r)
    nearest_c = np.rint(inv_c).astype(np.int64)
    nearest_r = np.rint(inv_r).astype(np.int64)
    valid = (nearest_r >= 0) & (nearest_r < rows) & (nearest_c >= 0) & (nearest_c < cols)

    out = np.zeros_like(component_mask, dtype=bool)
    sampled = np.zeros_like(valid, dtype=bool)
    sampled[valid] = component_mask[nearest_r[valid], nearest_c[valid]]
    out[min_r:max_r + 1, min_c:max_c + 1] = sampled
    return out


def apply_overlap_aware_transform(slice_2d, component_mask, transformed_mask, nii_path, axis, slice_idx, label):
    """
    单通道 label mask 无法同时存两个重叠类别。
    为了让上层标注移走后下层保持完整，这里在内存里记录被当前上层盖住的原像素值。
    """
    import numpy as np

    key = _overlap_key(nii_path, axis, slice_idx)
    backups = _overlap_backups.get(key, [])
    label = int(label)

    remaining_backups = []
    for entry in backups:
        top_mask = entry['top_mask']
        under_values = entry['under_values']
        if int(entry['top_label']) == label and np.any(top_mask & component_mask):
            restore_mask = top_mask & component_mask & (slice_2d == label)
            slice_2d[restore_mask] = under_values[restore_mask]
        else:
            remaining_backups.append(entry)

    under_values = slice_2d.copy()
    overlap_mask = transformed_mask & (slice_2d != 0) & (slice_2d != label)
    if np.any(overlap_mask):
        remaining_backups.append({
            'top_label': label,
            'top_mask': transformed_mask.copy(),
            'under_values': under_values,
        })

    old_visible_mask = component_mask & (slice_2d == label)
    slice_2d[old_visible_mask] = 0
    slice_2d[transformed_mask] = label

    if remaining_backups:
        _overlap_backups[key] = remaining_backups
    else:
        _overlap_backups.pop(key, None)
    return slice_2d


@bp.route('/nii_mask_transform_blob', methods=['POST'])
@login_required
def nii_mask_transform_blob():
    """
    原子化更新选中的 2D 连通域：后端从真实 NIfTI 中提取连通域，再进行平移/缩放/旋转。
    不能使用前端 overlay 轮廓写回，否则缩放后容易只保存断裂边缘。
    """
    import numpy as np
    data = request.get_json(silent=True) or {}
    nii_path  = data.get('nii_path', '').strip()
    axis      = data.get('axis', 'z')
    slice_idx = int(data.get('slice_idx', 0))
    vx        = int(data.get('vx', 0))
    vy        = int(data.get('vy', 0))
    vz        = int(data.get('vz', 0))
    label     = int(float(data.get('label', 1) or 1))
    canvas_w  = max(1.0, float(data.get('canvas_w', 512)))
    canvas_h  = max(1.0, float(data.get('canvas_h', 512)))
    center_cx = float(data.get('center_cx', canvas_w / 2.0))
    center_cy = float(data.get('center_cy', canvas_h / 2.0))
    translate_cx = float(data.get('translate_cx', 0.0))
    translate_cy = float(data.get('translate_cy', 0.0))
    scale = float(data.get('scale', 1.0))
    angle = float(data.get('angle', 0.0))

    if not nii_path or not os.path.isfile(nii_path):
        return jsonify({'error': 'NIfTI 文件不存在'}), 404
    if axis not in ('z', 'y', 'x'):
        return jsonify({'error': 'axis 参数无效'}), 400
    if label <= 0:
        return jsonify({'error': '选中标注类别无效'}), 400

    try:
        from medseg.utils.ct_utils import _get_cached_array, _load_sitk, clear_volume_cache

        arr = _get_cached_array(nii_path)
        Z, Y, X = arr.shape
        slice_idx = max(0, min(slice_idx, {'z': Z - 1, 'y': Y - 1, 'x': X - 1}[axis]))
        vx = max(0, min(vx, X - 1))
        vy = max(0, min(vy, Y - 1))
        vz = max(0, min(vz, Z - 1))

        seed_rc = find_seed_in_slice(arr, axis, slice_idx, vx, vy, vz, label, max_radius=96)
        if seed_rc is None:
            return jsonify({'error': f'未找到类别值为 {label} 的原始标注区域'}), 400

        if axis == 'z':
            slice_2d = arr[slice_idx, :, :]
            _push_slice_undo(nii_path, 'z', slice_idx, slice_2d)
            seed_r, seed_c = seed_rc
            rows, cols = Y, X
            arr[slice_idx, :, :] = slice_2d
        elif axis == 'y':
            slice_2d = arr[:, slice_idx, :]
            _push_slice_undo(nii_path, 'y', slice_idx, slice_2d)
            seed_r, seed_c = seed_rc
            rows, cols = Z, X
            arr[:, slice_idx, :] = slice_2d
        else:
            slice_2d = arr[:, :, slice_idx]
            _push_slice_undo(nii_path, 'x', slice_idx, slice_2d)
            seed_r, seed_c = seed_rc
            rows, cols = Z, Y
            arr[:, :, slice_idx] = slice_2d

        component_mask = connected_component_mask_2d(slice_2d, seed_r, seed_c, label)
        if component_mask is None or not np.any(component_mask):
            _undo_stacks[nii_path].pop()
            return jsonify({'error': f'未找到类别值为 {label} 的原始标注区域'}), 400

        if axis == 'z':
            center_r = center_cy / canvas_h * rows
            translate_r = translate_cy / canvas_h * rows
        else:
            center_r = (1.0 - (center_cy / canvas_h)) * rows
            translate_r = -(translate_cy / canvas_h * rows)
        center_c = center_cx / canvas_w * cols
        translate_c = translate_cx / canvas_w * cols
        transformed_mask = transform_component_mask_2d(
            component_mask,
            center_c,
            center_r,
            translate_c,
            translate_r,
            scale,
            angle,
        )
        if not np.any(transformed_mask):
            _undo_stacks[nii_path].pop()
            return jsonify({'error': '变换后标注区域为空，请减小移动或缩放幅度'}), 400

        slice_2d = apply_overlap_aware_transform(
            slice_2d,
            component_mask,
            transformed_mask,
            nii_path,
            axis,
            slice_idx,
            label,
        )
        if axis == 'z':
            arr[slice_idx, :, :] = slice_2d
        elif axis == 'y':
            arr[:, slice_idx, :] = slice_2d
        else:
            arr[:, :, slice_idx] = slice_2d

        sitk_lib = _load_sitk()
        ref = sitk_lib.ReadImage(nii_path)
        new_img = sitk_lib.GetImageFromArray(arr.astype(np.int16))
        new_img.CopyInformation(ref)
        sitk_lib.WriteImage(new_img, nii_path)
        clear_volume_cache(nii_path)

        return jsonify({'success': True, 'label': label})
    except Exception as e:
        logger.error(f"变换 CT 标注失败: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/nii_mask_commit', methods=['POST'])
@login_required
def nii_mask_commit():
    """
    CT 标注现已采用自动保存到真实标注文件，此接口保留兼容。
    """
    return jsonify({'success': True, 'message': '当前 CT 标注已自动保存到真实标注文件'})


@bp.route('/ct_annotations/<project_name>/<int:volume_id>')
@login_required
def ct_annotations(project_name, volume_id):
    """
    返回某切片上可见的结节标注（从 CTAnnotations 表的世界坐标投影到像素空间）。
    查询参数：
      axis  : z(轴状，默认) | y(冠状) | x(矢状)
      index : 切片索引 (0-based)
    返回 JSON: {annotations: [{label, px, py, r_px, diameter_mm, anno_id}]}
    其中 px/py 为切片像素中心，r_px 为像素半径
    """
    axis = request.args.get('axis', 'z').lower()
    try:
        index = int(request.args.get('index', 0))
    except (ValueError, TypeError):
        return jsonify({'error': '参数格式错误'}), 400

    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)

    # 读取体素间距和 origin
    import sqlite3
    spacing_x = spacing_y = spacing_z = 1.0
    origin_x = origin_y = origin_z = 0.0
    shape_x = shape_y = shape_z = 1
    try:
        db_path = os.path.join(project_path, 'config.db')
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                'SELECT spacing_x, spacing_y, spacing_z, shape_x, shape_y, shape_z FROM CTVolumes WHERE volume_id = ?',
                (volume_id,)
            ).fetchone()
            if row:
                spacing_x, spacing_y, spacing_z = float(row[0]), float(row[1]), float(row[2])
                shape_x, shape_y, shape_z = int(row[3]), int(row[4]), int(row[5])
    except Exception as e:
        logger.warning(f"读取体素间距失败 volume_id={volume_id}: {e}")

    # 获取 origin（从 SimpleITK）
    mhd_path = project.get_ct_volume_path(volume_id)
    if mhd_path and os.path.isfile(mhd_path):
        try:
            from medseg.utils.ct_utils import _load_sitk
            sitk = _load_sitk()
            reader = sitk.ImageFileReader()
            reader.SetFileName(mhd_path)
            reader.ReadImageInformation()
            origin = reader.GetOrigin()
            origin_x, origin_y, origin_z = float(origin[0]), float(origin[1]), float(origin[2])
        except Exception:
            pass

    # 读取所有标注（世界坐标）
    all_annos = project.get_ct_annotations_for_volume(volume_id)

    # 世界坐标 → 体素索引
    def world_to_voxel(cx, cy, cz):
        vx = (cx - origin_x) / spacing_x
        vy = (cy - origin_y) / spacing_y
        vz = (cz - origin_z) / spacing_z
        return vx, vy, vz  # ITK convention: vx→col(x), vy→row(y), vz→slice(z)

    # 过滤本切片可见的标注（在直径范围内）
    visible = []
    for a in all_annos:
        vx, vy, vz = world_to_voxel(a['coord_x'], a['coord_y'], a['coord_z'])
        diam = a['diameter_mm'] or 0
        # 沿切割轴的像素半径（单位：像素）
        if axis == 'z':
            slice_coord = vz
            r_along = diam / 2.0 / spacing_z if spacing_z > 0 else 0
            px_col, px_row = vx, vy   # 图像像素坐标 (col=x, row=y)
            r_px_display = diam / 2.0 / spacing_x if spacing_x > 0 else 0
        elif axis == 'y':
            slice_coord = vy
            r_along = diam / 2.0 / spacing_y if spacing_y > 0 else 0
            px_col, px_row = vx, (shape_z - 1) - vz
            r_px_display = diam / 2.0 / spacing_x if spacing_x > 0 else 0
        else:  # x
            slice_coord = vx
            r_along = diam / 2.0 / spacing_x if spacing_x > 0 else 0
            px_col, px_row = vy, (shape_z - 1) - vz
            r_px_display = diam / 2.0 / spacing_y if spacing_y > 0 else 0

        dist = abs(slice_coord - index)
        if r_along > 0 and dist > r_along:
            continue  # 不在这张切片的可见范围内
        if r_along == 0 and dist > 2:
            continue  # 直径未知时±2片宽度内显示

        # 投影后的圆半径（在当前切片上可见的圆截面半径）
        import math
        if r_along > 0 and dist < r_along:
            r_vis = math.sqrt(max(0, r_along**2 - dist**2))
        else:
            r_vis = r_px_display if r_px_display > 0 else 8  # 默认 8px

        visible.append({
            'anno_id': a['anno_id'],
            'label': a['label'],
            'px': round(px_col, 1),
            'py': round(px_row, 1),
            'r_px': round(r_vis, 1),
            'diameter_mm': diam,
        })

    return jsonify({'success': True, 'annotations': visible, 'axis': axis, 'index': index})


@bp.route('/update_ct_annotation', methods=['POST'])
@login_required
def update_ct_annotation():
    """更新单个结节的临床诊断属性（位置、实质性、危险程度、征象）"""
    data = request.get_json(silent=True) or {}
    project_name = data.get('project_name')
    anno_id = data.get('anno_id')
    location = data.get('location')
    texture = data.get('texture')
    risk_level = data.get('risk_level')
    signs = data.get('signs')  # 逗号分隔的征象列表

    if not project_name or anno_id is None:
        return jsonify({'success': False, 'error': '缺少必要参数'}), 400

    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'error': '项目不存在'}), 404

    try:
        import sqlite3
        db_path = os.path.join(project_path, 'config.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE CTAnnotations
                SET location = ?, texture = ?, risk_level = ?, signs = ?
                WHERE anno_id = ?
            ''', (location, texture, risk_level, signs, anno_id))
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"更新 CT 标注临床属性失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/ct_followup_history/<project_name>/<int:volume_id>')
@login_required
def ct_followup_history(project_name, volume_id):
    """
    根据 CT 文件名编号信息识别同一个人，并返回随访历史诊断记录。
    支持跨项目扫描，并且当数据库中无标注数据时自动解析 NIfTI 分割文件得出结节摘要。
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    current_vol = project.get_ct_volume(volume_id)
    if not current_vol:
        return jsonify({'error': '数据不存在'}), 404

    def get_patient_id(filename):
        name = os.path.splitext(filename)[0]
        if name.endswith('.nii'):
            name = name[:-4]
        if '_' in name:
            return name.split('_')[0]
        if '-' in name:
            parts = name.split('-')
            if len(parts) >= 3 and parts[0].lower() == 'lidc' and parts[1].lower() == 'idri':
                return f"LIDC-IDRI-{parts[2]}"
            return parts[0]
        if name.startswith("1.3.6.1.4.1.14519.5.2.1.6279.6001."):
            suffix = name.replace("1.3.6.1.4.1.14519.5.2.1.6279.6001.", "")
            if len(suffix) >= 2:
                return f"Patient_{suffix[:2]}"
            return "Patient_Unknown"
        return name

    curr_patient_id = get_patient_id(current_vol['name'])
    
    # 扫描 PROJECTS_FOLDER 下所有的项目目录
    all_projects = []
    for item in os.listdir(PROJECTS_FOLDER):
        item_path = os.path.join(PROJECTS_FOLDER, item)
        if os.path.isdir(item_path) and item != 'temp_chunks':
            db_path = os.path.join(item_path, 'config.db')
            if os.path.exists(db_path):
                all_projects.append(item)
                
    history = []
    
    for prj_name in all_projects:
        prj_path = os.path.join(PROJECTS_FOLDER, prj_name)
        prj = Project(prj_name, '', '', prj_path)
        try:
            vols = prj.get_ct_volumes()
            for vol in vols:
                vol_id = vol['volume_id']
                # 排除当前项目下当前正在查看的这一份
                if prj_name == project_name and vol_id == volume_id:
                    continue
                    
                pat_id = get_patient_id(vol['name'])
                if pat_id == curr_patient_id:
                    nodules_count = 0
                    max_diam = 0.0
                    
                    # 1. 优先尝试从数据库加载标注
                    annotations = prj.get_ct_annotations_for_volume(vol_id)
                    if annotations:
                        nodules_count = len(annotations)
                        max_diam = max([a.get('diameter_mm') or 0 for a in annotations] or [0])
                    else:
                        # 2. 如果数据库中无标注，但存在对应的分割文件，则对分割进行三维连通域提取计算结节
                        nii_path = vol.get('nii_path')
                        # 如果是相对路径，转换为绝对路径
                        if nii_path and not os.path.isabs(nii_path):
                            nii_path = os.path.join(prj_path, nii_path)
                            
                        if nii_path and os.path.isfile(nii_path):
                            try:
                                import numpy as np
                                from scipy import ndimage
                                from medseg.utils.ct_utils import _get_cached_array
                                arr = _get_cached_array(nii_path)
                                mask = arr != 0
                                if np.any(mask):
                                    labeled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
                                    nodules_count = count
                                    # 估算每个结节的最大外接圆直径 (mm)
                                    objects = ndimage.find_objects(labeled)
                                    spacing_z = float(vol.get('spacing_z') or 1.0)
                                    spacing_y = float(vol.get('spacing_y') or 1.0)
                                    spacing_x = float(vol.get('spacing_x') or 1.0)
                                    for slc in objects:
                                        if slc:
                                            z_len = (slc[0].stop - slc[0].start) * spacing_z
                                            y_len = (slc[1].stop - slc[1].start) * spacing_y
                                            x_len = (slc[2].stop - slc[2].start) * spacing_x
                                            max_diam = max(max_diam, z_len, y_len, x_len)
                            except Exception as e:
                                logger.warning(f"从随访历史 NIfTI 提取结节数据失败: {e}")
                    
                    added_at = vol.get('added_at') or ''
                    
                    # 生成诊断结论摘要
                    if nodules_count == 0:
                        summary = "未发现明显肺结节。"
                    else:
                        risk_level = "低危" if max_diam < 5 else ("中危" if max_diam < 10 else "高危")
                        summary = f"发现 {nodules_count} 处结节，最大结节分类为【{risk_level}】，最大径 {max_diam:.1f}mm。"
                    
                    history.append({
                        'project_name': prj_name,
                        'volume_id': vol_id,
                        'name': vol['name'],
                        'added_at': added_at,
                        'nodules_count': nodules_count,
                        'max_diameter': round(max_diam, 1),
                        'summary': summary
                    })
        except Exception as e:
            logger.warning(f"读取随访项目 {prj_name} 数据库失败: {e}")
            
    # 按照 added_at 排序
    history.sort(key=lambda x: x['added_at'] or '')
    
    return jsonify({
        'success': True,
        'patient_id': curr_patient_id,
        'history': history
    })


@bp.route('/ct_detection_summary/<project_name>/<int:volume_id>')
@login_required
def ct_detection_summary(project_name, volume_id):
    """
    返回 CT 体数据的类别检测摘要。
    包含自动解剖定位估算、风险评估和肺结节临床分期分析。
    """
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'error': '您没有权限访问该 CT 任务'}), 403
    project_classes = project.get_classes() or ['nodule']
    annotations = project.get_ct_annotations_for_volume(volume_id)
    source = 'annotation'
    findings = []

    # 提取体素空间定位
    spacing_x = float(volume.get('spacing_x') or 1.0) if volume else 1.0
    spacing_y = float(volume.get('spacing_y') or 1.0) if volume else 1.0
    spacing_z = float(volume.get('spacing_z') or 1.0) if volume else 1.0
    shape_x = int(volume.get('shape_x') or 512) if volume else 512
    shape_y = int(volume.get('shape_y') or 512) if volume else 512
    shape_z = int(volume.get('shape_z') or 1) if volume else 1
    origin_x = origin_y = origin_z = 0.0

    mhd_path = volume.get('absolute_path') if volume else None
    if mhd_path and os.path.isfile(mhd_path):
        try:
            from medseg.utils.ct_utils import _load_sitk
            sitk = _load_sitk()
            reader = sitk.ImageFileReader()
            reader.SetFileName(mhd_path)
            reader.ReadImageInformation()
            origin = reader.GetOrigin()
            origin_x, origin_y, origin_z = float(origin[0]), float(origin[1]), float(origin[2])
        except Exception:
            pass

    def classify_label(label, diameter_mm=0):
        text = (label or '').strip().lower()
        if not any(k in text for k in ('nodule', '结节', 'ggn', 'ggo', 'ground', 'solid', 'part', 'mixed', 'subsolid', 'calc', '磨玻璃', '磨砂', '实性', '部分实性', '亚实性', '钙化')):
            return label or '目标'
        if any(k in text for k in ('ground', 'ggn', 'ggo', '磨玻璃', '磨砂')):
            return '磨玻璃性肺结节'
        if any(k in text for k in ('part', 'mixed', 'subsolid', '混合', '部分实性', '亚实性')):
            return '部分实性肺结节'
        if any(k in text for k in ('calc', '钙化')):
            return '钙化结节'
        if any(k in text for k in ('solid', '实性')):
            return '实质性肺结节'
        if diameter_mm and diameter_mm < 6:
            return '微小肺结节'
        return label or '目标'

    def get_default_nodule_meta(x, y, z, diameter):
        mid_x = origin_x + (shape_x * spacing_x) / 2.0 if shape_x else 0.0
        mid_z = origin_z + (shape_z * spacing_z) / 2.0 if shape_z else 0.0
        
        side = "左肺" if x < mid_x else "右肺"
        if z > mid_z + 20:
            lobe = "上叶"
        elif z < mid_z - 20:
            lobe = "下叶"
        else:
            lobe = "中叶" if side == "右肺" else "上叶"
            
        if lobe == "下叶":
            seg = "后基底段"
        elif lobe == "上叶":
            seg = "前段"
        else:
            seg = "内侧段"
            
        location = f"{side}{lobe}{seg}"
        risk = "低危" if diameter < 5 else ("中危" if diameter < 10 else "高危")
        texture = "磨玻璃" if diameter < 6 else "实性"
        return location, risk, texture

    def merge_annotations_by_position(items, threshold_mm=3.0):
        clusters = []
        for anno in items:
            x = float(anno.get('coord_x') or 0)
            y = float(anno.get('coord_y') or 0)
            z = float(anno.get('coord_z') or 0)
            diameter = float(anno.get('diameter_mm') or 0)
            label = anno.get('label') or 'nodule'
            anno_id = anno.get('anno_id')
            location = anno.get('location')
            texture = anno.get('texture')
            risk_level = anno.get('risk_level')
            signs = anno.get('signs')

            matched = None
            for cluster in clusters:
                cx, cy, cz = cluster['center']
                dist = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5
                if dist <= threshold_mm:
                    matched = cluster
                    break

            if matched is None:
                clusters.append({
                    'center': [x, y, z],
                    'diameters': [diameter],
                    'labels': [label],
                    'count': 1,
                    'anno_id': anno_id,
                    'location': location,
                    'texture': texture,
                    'risk_level': risk_level,
                    'signs': signs
                })
            else:
                matched['count'] += 1
                matched['diameters'].append(diameter)
                matched['labels'].append(label)
                if not matched['location'] and location:
                    matched['location'] = location
                if not matched['texture'] and texture:
                    matched['texture'] = texture
                if not matched['risk_level'] and risk_level:
                    matched['risk_level'] = risk_level
                if not matched['signs'] and signs:
                    matched['signs'] = signs
                n = matched['count']
                matched['center'] = [
                    (matched['center'][0] * (n - 1) + x) / n,
                    (matched['center'][1] * (n - 1) + y) / n,
                    (matched['center'][2] * (n - 1) + z) / n,
                ]
        return clusters

    for cluster in merge_annotations_by_position(annotations):
        diameter = max(cluster['diameters'] or [0])
        label = next((v for v in cluster['labels'] if v), 'nodule')
        x, y, z = cluster['center']
        
        # 计算 Z 轴投影对应的切片索引
        slice_idx = int(round((z - origin_z) / spacing_z)) if spacing_z > 0 else 0
        slice_idx = max(0, min(slice_idx, shape_z - 1))
        
        loc_saved = cluster.get('location')
        risk_saved = cluster.get('risk_level')
        tex_saved = cluster.get('texture')
        signs_saved = cluster.get('signs') or ''
        
        def_loc, def_risk, def_tex = get_default_nodule_meta(x, y, z, diameter)
        
        location = loc_saved if loc_saved else def_loc
        risk_level = risk_saved if risk_saved else def_risk
        texture = tex_saved if tex_saved else def_tex

        findings.append({
            'anno_id': cluster.get('anno_id'),
            'type': classify_label(label, diameter),
            'diameter_mm': round(diameter, 1),
            'label': label,
            'slice_idx': slice_idx,
            'position': location,
            'coord_x': round(x, 1),
            'coord_y': round(y, 1),
            'coord_z': round(z, 1),
            'texture': texture,
            'risk_level': risk_level,
            'signs': signs_saved,
            'merged_annotations': cluster['count'],
        })

    nii_path = request.args.get('nii_path', '').strip()
    if not findings and nii_path:
        try:
            if os.path.isfile(nii_path) and nii_path.lower().endswith(('.nii', '.nii.gz')):
                import numpy as np
                from scipy import ndimage
                from medseg.utils.ct_utils import _get_cached_array, get_nii_metadata

                arr = _get_cached_array(nii_path)
                mask = arr != 0
                if np.any(mask):
                    labeled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
                    objects = ndimage.find_objects(labeled)
                    source = 'mask'
                    sx = spacing_x
                    sy = spacing_y
                    sz = spacing_z
                    ox = origin_x
                    oy = origin_y
                    oz = origin_z
                    try:
                        meta = get_nii_metadata(nii_path)
                        spacing = meta.get('spacing') or (spacing_x, spacing_y, spacing_z)
                        sx, sy, sz = [float(v) for v in spacing[:3]]
                        meta_origin = meta.get('origin') or (origin_x, origin_y, origin_z)
                        ox, oy, oz = [float(v) for v in meta_origin[:3]]
                    except Exception:
                        pass

                    for label_idx in range(1, count + 1):
                        slc = objects[label_idx - 1]
                        if not slc:
                            continue
                        component = labeled[slc] == label_idx
                        raw_vals = arr[slc][component]
                        raw_vals = raw_vals[raw_vals > 0]
                        label_value = int(np.bincount(raw_vals.astype(int)).argmax()) if raw_vals.size else 1
                        class_label = project_classes[label_value - 1] if 1 <= label_value <= len(project_classes) else f'class_{label_value}'
                        z_len = (slc[0].stop - slc[0].start) * sz
                        y_len = (slc[1].stop - slc[1].start) * sy
                        x_len = (slc[2].stop - slc[2].start) * sx
                        diameter = max(x_len, y_len, z_len)
                        
                        center_z_idx = (slc[0].start + slc[0].stop - 1) / 2.0
                        center_y_idx = (slc[1].start + slc[1].stop - 1) / 2.0
                        center_x_idx = (slc[2].start + slc[2].stop - 1) / 2.0

                        center_z = oz + center_z_idx * sz
                        center_y = oy + center_y_idx * sy
                        center_x = ox + center_x_idx * sx
                        
                        def_loc, def_risk, def_tex = get_default_nodule_meta(center_x, center_y, center_z, diameter)
                        slice_idx = int(round(center_z_idx))

                        findings.append({
                            'anno_id': None,
                            'type': classify_label(class_label, diameter),
                            'diameter_mm': round(float(diameter), 1),
                            'label': class_label,
                            'label_value': label_value,
                            'slice_idx': slice_idx,
                            'position': def_loc,
                            'coord_x': round(float(center_x), 1),
                            'coord_y': round(float(center_y), 1),
                            'coord_z': round(float(center_z), 1),
                            'texture': def_tex,
                            'risk_level': def_risk,
                            'signs': '',
                        })
        except Exception as e:
            logger.warning(f"计算 CT 检测摘要失败 volume_id={volume_id}: {e}")

    type_counts = {}
    label_counts = {}
    for item in findings:
        type_counts[item['type']] = type_counts.get(item['type'], 0) + 1
        key = item.get('label') or '未分类'
        label_counts[key] = label_counts.get(key, 0) + 1

    max_diameter = max([n['diameter_mm'] for n in findings if n.get('diameter_mm')] or [0])
    if not findings:
        advice = '提示：当前 CT 未检测到肺结节。建议年度低剂量螺旋 CT (LDCT) 定期复查随访。'
    else:
        high_risk_count = len([f for f in findings if f.get('risk_level') == '高危'])
        med_risk_count = len([f for f in findings if f.get('risk_level') == '中危'])
        low_risk_count = len([f for f in findings if f.get('risk_level') == '低危'])
        
        solid_count = len([f for f in findings if f.get('texture') == '实性'])
        ggn_count = len([f for f in findings if f.get('texture') == '磨玻璃'])
        subsolid_count = len([f for f in findings if f.get('texture') == '亚实性'])
        
        advice = f"诊断意见：双肺共检出 {len(findings)} 处肺结节。其中 "
        
        if high_risk_count > 0:
            high_locs = list(set([f.get('position') for f in findings if f.get('risk_level') == '高危' and f.get('position')]))
            loc_str = "、".join(high_locs[:3])
            if len(high_locs) > 3:
                loc_str += "等"
            advice += f"{loc_str}发现危险程度为【高危】的肺结节（最大径约 {max_diameter:.1f}mm），影像学表现存在占位及倾向性，强烈建议临床结合增强扫描、肿瘤标志物或穿刺活检评估排除恶性病变。"
        elif med_risk_count > 0:
            med_locs = list(set([f.get('position') for f in findings if f.get('risk_level') == '中危' and f.get('position')]))
            loc_str = "、".join(med_locs[:3])
            if len(med_locs) > 3:
                loc_str += "等"
            advice += f"{loc_str}结节大小介于 5-10mm，危险评估为【中危】，建议 3 个月内复查薄层 HRCT 随访，密切监视结节形态、大小及密度变化。"
        else:
            low_locs = list(set([f.get('position') for f in findings if f.get('risk_level') == '低危' and f.get('position')]))
            loc_str = "、".join(low_locs[:3])
            if len(low_locs) > 3:
                loc_str += "等"
            advice += f"结节（如{loc_str}）直径均小于 5mm，评估为【低危】，多考虑炎性肉芽肿或陈旧增殖灶等良性病变。建议 6-12 个月复查低剂量 CT 随访。"
            
        if ggn_count > 0 or subsolid_count > 0:
            advice += " 提醒：检出含有磨玻璃或混合亚实性成分结节，这类结节有早期腺癌演变风险，建议适当缩短随访周期，观察有无实性成分增加。"
        elif solid_count > 0:
            advice += " 结节以实性成分为主，随访时应注意结节径线增长速度。"

    return jsonify({
        'success': True,
        'count': len(findings),
        'types': [{'name': k, 'count': v} for k, v in sorted(type_counts.items())],
        'labels': [{'name': k, 'count': v} for k, v in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))],
        'findings': findings[:20],
        'max_diameter_mm': max_diameter,
        'source': source,
        'advice': advice,
    })


# ==============================================================
# MedSAM2 肺结节自动分割接口（预留存根，模型部署后接通）
# ==============================================================

def _medsam2_available() -> bool:
    """检查 MedSAM2 GPU 服务是否可用"""
    try:
        resp = requests.get(
            f"{_medsam2_service_url()}/health",
            timeout=max(3, _medsam2_service_timeout()),
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        return bool(data.get('available'))
    except Exception:
        return False


@bp.route('/medsam2/status')
@login_required
def medsam2_status():
    """
    返回 MedSAM2 服务状态。
    前端用此接口判断「一键分割」按钮是否可点击。
    """
    if _medsam2_available():
        try:
            resp = requests.get(
                f"{_medsam2_service_url()}/health",
                timeout=max(3, _medsam2_service_timeout()),
            )
            data = resp.json()
            return jsonify({
                'available': True,
                'message': 'MedSAM2 GPU 服务就绪',
                'device': data.get('device'),
                'checkpoint': data.get('checkpoint'),
                'config': data.get('config'),
            })
        except Exception:
            return jsonify({'available': True, 'message': 'MedSAM2 服务就绪'})
    return jsonify({
        'available': False,
        'message': 'MedSAM2 GPU 服务未启动，请先启动 start_medsam2_service.sh'
    })


@bp.route('/medsam2/segment/<project_name>/<int:volume_id>', methods=['POST'])
@login_required
def medsam2_segment(project_name, volume_id):
    """
    触发 MedSAM2 对整个 CT 体数据做肺结节自动分割。
    请求体（JSON，可选）：
      {
        "box_prompt":  [x1,y1,z1, x2,y2,z2],  # 可选 3D bounding box 提示
        "point_prompt": [[x,y,z,label], ...],  # 可选点提示 (label: 1=前景 0=背景)
        "output_path": "/absolute/path/output.nii.gz"  # 可选，默认自动生成
      }
    返回：
      {
        "success": true,
        "nii_path": "/path/to/output_mask.nii.gz",
        "task_id": "uuid",   # 异步任务ID（推理可能耗时）
        "message": "..."
      }
    ---
    ⚠️  当前为预留存根，MedSAM2 模型部署后替换 TODO 部分即可接通。
    接通步骤：
      1. 安装并启动 MedSAM2 推理服务（建议 FastAPI，监听 localhost:7001）
      2. 将 _medsam2_available() 改为实际的心跳检查
      3. 实现下方 TODO 块：发送请求到推理服务，接收 .nii.gz 路径
      4. 调用 project.add_ct_volume 的 nii_path 参数更新数据库
    """
    if not _medsam2_available():
        return jsonify({
            'success': False,
            'available': False,
            'message': 'MedSAM2 GPU 服务未启动，当前无法执行 CT 自动分割。',
            'docs': '请先启动 start_medsam2_service.sh'
        }), 503
    project_path = os.path.join(_projects_folder(), secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'message': '项目不存在'}), 404

    project = Project(project_name, '', '', project_path)
    volume = project.get_ct_volume(volume_id)
    if not volume:
        return jsonify({'success': False, 'message': 'CT 体数据不存在'}), 404
    if not _user_can_access_ct_volume(project_name, current_user, volume):
        return jsonify({'success': False, 'message': '您没有权限访问该 CT 任务'}), 403

    vol_path = project.get_ct_volume_path(volume_id)
    if not vol_path or not os.path.isfile(vol_path):
        return jsonify({'success': False, 'message': 'CT 体数据文件不存在'}), 404

    data = request.get_json(silent=True) or {}
    output_path = str(data.get('output_path') or '').strip() or _build_default_medsam2_output_path(project, volume, volume_id)
    points, boxes = _build_medsam2_prompts(project, volume_id, vol_path, data)
    prompt_mask_path = str(data.get('prompt_mask_path') or '').strip()
    prompt_mask_axis = str(data.get('prompt_mask_axis') or 'z').strip().lower()
    prompt_mask_slice_idx = data.get('prompt_mask_slice_idx')
    use_mask_prompt = bool(prompt_mask_path)
    if use_mask_prompt and (prompt_mask_slice_idx is None or str(prompt_mask_slice_idx).strip() == ''):
        return jsonify({'success': False, 'message': '缺少手工掩码所在的切片索引'}), 400

    if use_mask_prompt:
        if not os.path.isfile(prompt_mask_path):
            return jsonify({'success': False, 'message': '当前掩码文件不存在，请先初始化并绘制手工掩码'}), 400
        if prompt_mask_axis != 'z':
            return jsonify({'success': False, 'message': '当前仅支持在轴状面完成手工粗标后执行 AI 补全'}), 400
    if not points and not boxes and not use_mask_prompt:
        return jsonify({
            'success': False,
            'message': '当前没有可用于 MedSAM2 的提示信息。请先在轴状面手工粗标病灶后再执行 AI 标注。'
        }), 400

    label_value = int(data.get('label_value') or 1)
    ww = float(data.get('ww') or 1500.0)
    wl = float(data.get('wl') or -600.0)
    existing_mask_path = str(volume.get('nii_path') or '').strip()
    if use_mask_prompt:
        if not existing_mask_path:
            return jsonify({
                'success': False,
                'message': '当前 CT 体数据还没有绑定掩码文件，请先初始化当前体数据的掩码后再执行 AI 标注。'
            }), 400
        if os.path.abspath(prompt_mask_path) != os.path.abspath(existing_mask_path):
            return jsonify({
                'success': False,
                'message': '当前手工掩码与所选 CT 体数据不匹配，请重新加载该体数据自己的掩码后再执行 AI 标注。'
            }), 400
        try:
            if not _has_mask_label_on_slice(prompt_mask_path, prompt_mask_axis, int(prompt_mask_slice_idx), label_value):
                return jsonify({
                    'success': False,
                    'message': '当前轴状面还没有这个类别的手工掩码，请先粗标病灶后再执行 AI 标注。'
                }), 400
        except Exception as exc:
            logger.warning("Failed to validate MedSAM2 prompt mask project=%s volume=%s error=%s", project_name, volume_id, exc)
            return jsonify({'success': False, 'message': f'检查手工掩码失败: {exc}'}), 400

    try:
        resp = requests.post(
            f"{_medsam2_service_url()}/segment",
            json={
                'volume_path': vol_path,
                'output_path': output_path,
                'points': points,
                'boxes': boxes,
                'prompt_mask_path': prompt_mask_path or None,
                'prompt_mask_axis': prompt_mask_axis,
                'prompt_mask_slice_idx': int(prompt_mask_slice_idx) if use_mask_prompt else None,
                'prompt_mask_label': label_value if use_mask_prompt else None,
                'label_value': label_value,
                'ww': ww,
                'wl': wl,
            },
            timeout=max(30, _medsam2_service_timeout()),
        )
        result = resp.json()
    except Exception as exc:
        logger.error("Failed to submit MedSAM2 task project=%s volume=%s error=%s", project_name, volume_id, exc)
        return jsonify({'success': False, 'message': f'提交 MedSAM2 任务失败: {exc}'}), 502

    if resp.status_code >= 400 or not result.get('success'):
        return jsonify({'success': False, 'message': result.get('detail') or result.get('message') or 'MedSAM2 推理服务拒绝请求'}), 502

    task_id = str(result.get('task_id') or '')
    if not task_id:
        return jsonify({'success': False, 'message': 'MedSAM2 服务未返回任务 ID'}), 502

    _medsam2_store_task(task_id, {
        'project_name': project_name,
        'project_path': project_path,
        'volume_name': volume.get('name') or f'ct_volume_{volume_id}',
        'volume_id': volume_id,
        'output_path': output_path,
        'existing_mask_path': existing_mask_path,
        'requested_label_value': label_value,
        'submitted_at': time.time(),
    })
    return jsonify({
        'success': True,
        'task_id': task_id,
        'nii_path': output_path,
        'message': 'MedSAM2 GPU 推理任务已提交',
    })


@bp.route('/medsam2/segment_status/<task_id>')
@login_required
def medsam2_segment_status(task_id):
    """
    查询异步推理任务状态（MedSAM2 推理 CT 数据可能需要几十秒）。
    返回：{status: 'pending'|'running'|'done'|'error', progress: 0~100, nii_path: ...}
    ---
    ⚠️  预留存根，接通后实现轮询推理服务的任务队列。
    """
    if not _medsam2_available():
        return jsonify({'status': 'unavailable', 'message': 'MedSAM2 GPU 服务未启动'}), 503

    try:
        resp = requests.get(
            f"{_medsam2_service_url()}/tasks/{task_id}",
            timeout=max(5, _medsam2_service_timeout()),
        )
        result = resp.json()
    except Exception as exc:
        logger.error("Failed to query MedSAM2 task status task_id=%s error=%s", task_id, exc)
        return jsonify({'status': 'error', 'message': f'查询 MedSAM2 任务状态失败: {exc}', 'progress': 100}), 502

    if resp.status_code == 404:
        return jsonify({'status': 'error', 'message': 'MedSAM2 任务不存在', 'progress': 100}), 404
    if resp.status_code >= 400:
        return jsonify({'status': 'error', 'message': result.get('detail') or 'MedSAM2 服务异常', 'progress': 100}), 502

    task_meta = _medsam2_get_task(task_id)
    if task_meta:
        project_name = str(task_meta.get('project_name') or '')
        project_path = str(task_meta.get('project_path') or '')
        if project_name and project_path and os.path.exists(project_path):
            project = Project(project_name, '', '', project_path)
            volume = project.get_ct_volume(int(task_meta.get('volume_id') or 0))
            if not _user_can_access_ct_volume(project_name, current_user, volume):
                return jsonify({'status': 'error', 'message': '您没有权限访问该 CT 任务', 'progress': 100}), 403
    status = str(result.get('status') or 'pending')
    nii_path = result.get('nii_path') or (task_meta or {}).get('output_path')

    if status == 'done' and task_meta and nii_path and os.path.isfile(nii_path):
        try:
            project = Project(task_meta['project_name'], '', '', task_meta['project_path'])
            final_path = nii_path
            volume_id = int(task_meta['volume_id'])
            volume = project.get_ct_volume(volume_id) or {}
            existing_mask_path = str(task_meta.get('existing_mask_path') or '').strip()
            managed_mask_path = None

            if existing_mask_path and os.path.isfile(existing_mask_path):
                _, managed_mask_path, _ = _ensure_project_managed_mask(
                    project,
                    volume_id,
                    preferred_nii_path=existing_mask_path,
                )
            else:
                managed_mask_path = _build_ct_mask_storage_path(
                    task_meta['project_path'],
                    volume.get('name') or task_meta.get('volume_name') or f'ct_volume_{volume_id}',
                    volume_id,
                    nii_path,
                )

            backup_path = _build_ct_undo_backup_path(
                task_meta['project_path'],
                task_meta.get('volume_name') or volume.get('name'),
                volume_id,
            )

            try:
                if os.path.isfile(managed_mask_path):
                    shutil.copy2(managed_mask_path, backup_path)
                    final_path = _merge_ct_masks(managed_mask_path, nii_path, output_path=managed_mask_path)
                else:
                    _create_empty_mask_like(nii_path, backup_path)
                    os.makedirs(os.path.dirname(managed_mask_path), exist_ok=True)
                    shutil.copy2(nii_path, managed_mask_path)
                    final_path = managed_mask_path
                _push_volume_restore_undo(final_path, backup_path)
                if os.path.exists(nii_path) and os.path.abspath(nii_path) != os.path.abspath(final_path):
                    os.remove(nii_path)
            except Exception as merge_exc:
                try:
                    os.remove(backup_path)
                except OSError:
                    pass
                logger.warning("Failed to merge MedSAM2 output into existing mask task_id=%s error=%s", task_id, merge_exc)
                final_path = nii_path

            project.update_ct_volume_nii(volume_id, final_path)
            nii_path = final_path
        except Exception as exc:
            logger.warning("Failed to persist MedSAM2 output path task_id=%s error=%s", task_id, exc)

    return jsonify({
        'status': status,
        'progress': int(result.get('progress') or 0),
        'message': result.get('message') or '',
        'nii_path': nii_path,
    })
