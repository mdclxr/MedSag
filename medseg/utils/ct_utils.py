"""
CT 体数据工具模块
支持 MHD/RAW 和 NIfTI (.nii / .nii.gz) 格式的读取、元数据解析和切片按需渲染
依赖：SimpleITK, numpy, Pillow

性能优化：首次读取后将 numpy 体数组缓存在内存中（最多 MAX_CACHED_VOLUMES 个体数据），
后续切片请求直接从内存取，消除每次请求的全文件 IO 卡顿。
"""

import os
import io
import logging
import threading
import collections
import numpy as np

logger = logging.getLogger(__name__)

# ===== 内存体数据缓存 =====
MAX_CACHED_VOLUMES = 5   # 最多同时缓存 5 个体数据
_cache_lock = threading.Lock()
# OrderedDict 用作 LRU：key = file_path, value = np.ndarray (z,y,x)
_volume_cache: collections.OrderedDict = collections.OrderedDict()
# 同步存储每个体数据的值域信息，避免每次切片都扫描整个数组
_volume_range_cache: dict = {}   # file_path -> (arr_min, arr_max)

_nii_file_lock = threading.RLock()
_sitk_patched = False
_EST_8BIT_CT_HU_MIN = -1000.0
_EST_8BIT_CT_HU_MAX = 400.0

# NIfTI 文件扩展名集合
_NII_EXTS = {'.nii', '.nii.gz'}


def _is_nii(path: str) -> bool:
    """判断路径是否为 NIfTI 格式"""
    p = path.lower()
    return p.endswith('.nii') or p.endswith('.nii.gz')


def _get_cached_array(file_path: str) -> np.ndarray:
    """
    获取体数据 numpy 数组（float32，HU 值 / 或 mask 整数值）。
    自动识别 MHD 和 NIfTI 格式，首次调用读取并缓存，后续直接返回内存数组。
    线程安全。
    """
    with _cache_lock:
        if file_path in _volume_cache:
            _volume_cache.move_to_end(file_path)
            return _volume_cache[file_path]

    logger.info(f"CT 体数组缓存未命中，读取文件: {file_path}")
    sitk = _load_sitk()
    image = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)

    arr_min = float(arr.min())
    arr_max = float(arr.max())

    with _cache_lock:
        if file_path not in _volume_cache:
            while len(_volume_cache) >= MAX_CACHED_VOLUMES:
                evicted, _ = _volume_cache.popitem(last=False)
                _volume_range_cache.pop(evicted, None)
                logger.info(f"CT 缓存淘汰: {evicted}")
            _volume_cache[file_path] = arr
            _volume_range_cache[file_path] = (arr_min, arr_max)
        _volume_cache.move_to_end(file_path)

    logger.info(f"CT 体数组已缓存: {file_path}, shape={arr.shape}, range=[{arr_min:.1f}, {arr_max:.1f}]")
    return arr


# 保留旧别名以保持向后兼容
_get_cached_mhd_array = _get_cached_array


def clear_volume_cache(file_path: str = None):
    """手动清除缓存（可选指定路径，支持 MHD 和 NIfTI）"""
    with _cache_lock:
        if file_path:
            _volume_cache.pop(file_path, None)
            _volume_range_cache.pop(file_path, None)
        else:
            _volume_cache.clear()
            _volume_range_cache.clear()


def _is_explicit_8bit_ct_image(file_path: str, arr_min: float, arr_max: float) -> bool:
    """
    判断当前体数据是否是明确的 8-bit 预处理 CT 图像。

    只有在以下条件同时满足时，才绕过真实 CT 的 WW/WC 窗控：
      1. 文件是 NIfTI
      2. 文件名明确带有 `_img.nii` / `_img.nii.gz` 约定
      3. 体数据值域落在 8-bit 灰度范围内

    这样可以避免把原始 HU 的 NIfTI/MHD 误判为普通 0~255 图像，
    导致“软组织 / 肺窗 / 骨窗”切换看起来没有变化。
    """
    if not _is_nii(file_path):
        return False

    base = os.path.basename(file_path).lower()
    is_tagged_img = base.endswith('_img.nii') or base.endswith('_img.nii.gz')
    return is_tagged_img and arr_min >= 0.0 and arr_max <= 255.0


def _is_lung_window_preset(ww: float, wc: float) -> bool:
    """判断是否为默认肺窗预设。"""
    return abs(float(ww) - 1500.0) < 1e-3 and abs(float(wc) - (-600.0)) < 1e-3


def _render_8bit_lung_like_view(slice_2d: np.ndarray) -> np.ndarray:
    """
    针对 `_img.nii(.gz)` 的 8-bit CT，生成更接近原图观感的“肺窗观感”。

    这类数据往往已经被预处理成适合肺部查看的灰度分布，
    再做一次固定 HU 反推会让整幅图发灰；而过强的黑场拉伸又会让图像偏黑。
    因此这里改为：
      - 以原始 8-bit 灰度为主
      - 只做非常轻的亮度/对比增强
      - 保持整体风格接近导入前的原图效果，只让肺实质细节更易观察
    """
    src = slice_2d.astype(np.float32)
    norm = np.clip(src / 255.0, 0.0, 1.0)

    # 轻微提亮中暗部，让肺纹理更清楚，但不改变整体风格。
    boosted = np.power(norm, 0.88)
    blended = norm * 0.78 + boosted * 0.22

    out = np.round(np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[slice_2d <= 0] = 0
    return out


def _apply_ct_window(slice_2d: np.ndarray,
                     file_path: str,
                     ww: float,
                     wc: float,
                     arr_min: float,
                     arr_max: float) -> np.ndarray:
    """
    将单张切片映射为 8-bit 灰度。

    - 原始 HU 体数据：直接按 WW/WC 做标准 CT 窗控
    - `_img.nii(.gz)` 8-bit 预处理图像：按肺 CT 常见预处理区间
      [-1000, 400] 近似反推为 HU 后再做窗控

    说明：
    8-bit `_img` 数据已经丢失原始 HU，无法做到医学意义上的“完全真实”窗控；
    这里做的是与预设语义一致的近似重建，让软组织窗/肺窗/骨窗在该类数据上
    仍然产生稳定且明显的可视差异。
    """
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    denom = max(hi - lo, 1e-6)

    if _is_explicit_8bit_ct_image(file_path, arr_min, arr_max):
        if _is_lung_window_preset(ww, wc):
            return _render_8bit_lung_like_view(slice_2d)

        # 8-bit CT 图像通常是从固定 HU 区间裁剪后再缩放到 [0,255]。
        # 这里按肺结节 CT 的常见预处理区间做近似逆映射。
        hu = (
            slice_2d.astype(np.float32) / 255.0
            * (_EST_8BIT_CT_HU_MAX - _EST_8BIT_CT_HU_MIN)
            + _EST_8BIT_CT_HU_MIN
        )
        windowed = np.clip(hu, lo, hi)
        out = ((windowed - lo) / denom * 255).astype(np.uint8)
        out[slice_2d <= 0] = 0
        return out

    windowed = np.clip(slice_2d, lo, hi)
    return ((windowed - lo) / denom * 255).astype(np.uint8)





def _load_sitk():
    """延迟导入 SimpleITK，避免无 CT 功能时的启动依赖"""
    global _sitk_patched
    try:
        import SimpleITK as sitk
        if not _sitk_patched:
            orig_read = sitk.ReadImage
            orig_write = sitk.WriteImage
            
            def locked_read(*args, **kwargs):
                with _nii_file_lock:
                    return orig_read(*args, **kwargs)
                    
            def locked_write(*args, **kwargs):
                with _nii_file_lock:
                    return orig_write(*args, **kwargs)
                    
            sitk.ReadImage = locked_read
            sitk.WriteImage = locked_write
            _sitk_patched = True
        return sitk
    except ImportError:
        raise ImportError(
            "SimpleITK 未安装。请运行: pip install SimpleITK"
        )


def validate_mhd_pair(mhd_path: str) -> bool:
    """
    检查 MHD 文件是否有配套的 RAW 文件。
    MHD 文件中 ElementDataFile 字段指向 RAW 文件。
    """
    if not os.path.isfile(mhd_path):
        return False
    raw_ref = None
    try:
        with open(mhd_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().lower().startswith('elementdatafile'):
                    raw_ref = line.split('=', 1)[1].strip()
                    break
    except Exception as e:
        logger.error(f"读取 MHD 文件失败 {mhd_path}: {e}")
        return False

    if not raw_ref or raw_ref.upper() == 'LOCAL':
        # LOCAL 表示数据内嵌在 MHD 中
        return True
    raw_path = os.path.join(os.path.dirname(mhd_path), raw_ref)
    return os.path.isfile(raw_path)


def parse_volume_metadata(file_path: str) -> dict:
    """
    解析 MHD 或 NIfTI 文件，返回体数据元数据。
    返回字段：shape (z,y,x), spacing (z,y,x), origin, direction
    """
    sitk = _load_sitk()
    try:
        with _nii_file_lock:
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.ReadImageInformation()
            size = reader.GetSize()        # (x, y, z)
            spacing = reader.GetSpacing()  # (x, y, z) in mm
            origin = reader.GetOrigin()
            direction = reader.GetDirection()
        return {
            'shape_x': int(size[0]),
            'shape_y': int(size[1]),
            'shape_z': int(size[2]),
            'spacing_x': float(spacing[0]),
            'spacing_y': float(spacing[1]),
            'spacing_z': float(spacing[2]),
            'origin': list(origin),
            'direction': list(direction),
        }
    except Exception as e:
        logger.error(f"解析体数据元数据失败 {file_path}: {e}")
        raise


# 保留旧函数名以兼容现有代码
def parse_mhd_metadata(mhd_path: str) -> dict:
    return parse_volume_metadata(mhd_path)


def get_slice_jpeg(mhd_path: str, axis: str, index: int,
                   ww: float = 1500.0, wc: float = -600.0,
                   quality: int = 85,
                   spacing_x: float = 1.0,
                   spacing_y: float = 1.0,
                   spacing_z: float = 1.0) -> bytes:
    """
    从体数据中提取指定切片，返回 JPEG 字节流。
    支持 MHD/RAW（HU 值）和 NIfTI。

    窗口策略：
      - 对原始 CT（MHD/RAW、HU-NIfTI）按 WW/WC 做真实窗宽窗位映射
      - 仅对明确命名为 `_img.nii/.nii.gz` 且值域位于 [0,255] 的 8-bit
        预处理图像，才直接按原灰度范围显示

    spacing_* 单位 mm，用于冠状面/矢状面宽高比校正。
    """
    from PIL import Image

    arr = _get_cached_array(mhd_path)

    axis = axis.lower()
    if axis == 'z':
        index = max(0, min(index, arr.shape[0] - 1))
        slice_2d = arr[index, :, :]
    elif axis == 'y':
        index = max(0, min(index, arr.shape[1] - 1))
        slice_2d = arr[:, index, :][::-1, :]
    elif axis == 'x':
        index = max(0, min(index, arr.shape[2] - 1))
        slice_2d = arr[:, :, index][::-1, :]
    else:
        raise ValueError(f"axis 参数无效: {axis}，应为 'x'/'y'/'z'")

    # ===== 自动检测值域，选择窗口策略 =====
    # 优先从缓存读取，避免全量扫描
    with _cache_lock:
        _range = _volume_range_cache.get(mhd_path)
    if _range:
        arr_min, arr_max = _range
    else:
        arr_min = float(arr.min())
        arr_max = float(arr.max())

    slice_2d = _apply_ct_window(
        slice_2d,
        mhd_path,
        ww=ww,
        wc=wc,
        arr_min=arr_min,
        arr_max=arr_max,
    )

    pil_img = Image.fromarray(slice_2d, mode='L')

    # ===== 体素间距比例校正 =====
    if axis == 'y' and spacing_x > 0:
        scale = spacing_z / spacing_x
        if abs(scale - 1.0) > 0.02:
            w, h = pil_img.size
            new_h = max(1, int(round(h * scale)))
            pil_img = pil_img.resize((w, new_h), Image.LANCZOS)
    elif axis == 'x' and spacing_y > 0:
        scale = spacing_z / spacing_y
        if abs(scale - 1.0) > 0.02:
            w, h = pil_img.size
            new_h = max(1, int(round(h * scale)))
            pil_img = pil_img.resize((w, new_h), Image.LANCZOS)

    pil_img = pil_img.convert('RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality, optimize=False, subsampling=0)
    buf.seek(0)
    return buf.getvalue()



def get_slice_png(mhd_path: str, axis: str, index: int,
                  ww: float = 1500.0, wc: float = -600.0) -> bytes:
    """PNG 版本，保留备用。总一般情况下建议使用 get_slice_jpeg() 代替。"""
    from PIL import Image

    arr = _get_cached_array(mhd_path)

    axis = axis.lower()
    if axis == 'z':
        index = max(0, min(index, arr.shape[0] - 1))
        slice_2d = arr[index, :, :]
    elif axis == 'y':
        index = max(0, min(index, arr.shape[1] - 1))
        slice_2d = arr[:, index, :][::-1, :]
    elif axis == 'x':
        index = max(0, min(index, arr.shape[2] - 1))
        slice_2d = arr[:, :, index][::-1, :]
    else:
        raise ValueError(f"axis 参数无效: {axis}，应为 'x'/'y'/'z'")

    with _cache_lock:
        _range = _volume_range_cache.get(mhd_path)
    if _range:
        arr_min, arr_max = _range
    else:
        arr_min = float(arr.min())
        arr_max = float(arr.max())

    slice_2d = _apply_ct_window(
        slice_2d,
        mhd_path,
        ww=ww,
        wc=wc,
        arr_min=arr_min,
        arr_max=arr_max,
    )

    pil_img = Image.fromarray(slice_2d, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


def generate_preview_png(volume_path: str, save_path: str,
                         ww: float = 1500.0, wc: float = -600.0) -> bool:
    """
    生成中间层（Z轴中间切片）的 PNG 预览图，保存到 save_path。
    用于项目卡片缩略图。支持 MHD 和 NIfTI (.nii/.nii.gz) 格式。
    """
    try:
        meta = parse_volume_metadata(volume_path)
        mid_z = meta['shape_z'] // 2
        png_bytes = get_slice_png(volume_path, 'z', mid_z, ww=ww, wc=wc)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(png_bytes)
        return True
    except Exception as e:
        logger.error(f"生成预览图失败 {volume_path}: {e}")
        return False


# ============================================================
# NIfTI (.nii / .nii.gz) 分割掩码切片渲染
# ============================================================

# 默认掩码颜色表（标签值 -> RGBA），最多支持 8 个不同标签
_LABEL_COLORS = [
    (255, 100,  50, 180),   # 1 - 橙红（肺结节）
    ( 50, 200, 100, 180),   # 2 - 绿
    ( 50, 130, 255, 180),   # 3 - 蓝
    (220,  50, 220, 180),   # 4 - 紫
    (255, 220,  50, 180),   # 5 - 黄
    ( 50, 220, 220, 180),   # 6 - 青
    (255, 130, 200, 180),   # 7 - 粉
    (160, 255, 100, 180),   # 8 - 黄绿
]


def _extract_slice(arr: np.ndarray, axis: str, index: int):
    """从 (z,y,x) 体数组中提取 2D 切片，返回 (slice_2d, clamped_index)"""
    axis = axis.lower()
    if axis == 'z':
        index = max(0, min(index, arr.shape[0] - 1))
        return arr[index, :, :], index
    elif axis == 'y':
        index = max(0, min(index, arr.shape[1] - 1))
        return arr[:, index, :][::-1, :], index
    elif axis == 'x':
        index = max(0, min(index, arr.shape[2] - 1))
        return arr[:, :, index][::-1, :], index
    else:
        raise ValueError(f"axis 参数无效: {axis}，应为 'x'/'y'/'z'")


def get_nii_mask_slice_png(nii_path: str, axis: str, index: int,
                           spacing_x: float = 1.0,
                           spacing_y: float = 1.0,
                           spacing_z: float = 1.0,
                           alpha: int = 180,
                           color_rgb: tuple = None) -> bytes:
    """
    从 NIfTI 分割掩码中提取指定切片，返回 RGBA PNG 字节流。
    背景（值=0）为完全透明，前景（≥1）按标签值着色。
    前端可将此 PNG 叠加在 CT 灰度图上，实现分割结果可视化。

    参数：
        nii_path   : .nii 或 .nii.gz 文件的绝对路径
        axis       : 'z'(轴状) | 'y'(冠状) | 'x'(矢状)
        index      : 切片索引 (0-based)
        spacing_*  : 体素间距 (mm)，用于等比校正
        alpha      : 掩码不透明度 0~255（默认 180）
        color_rgb  : 强制指定的前景 RGB 颜色元组，例如 (16, 185, 129)
    返回：
        RGBA PNG 字节流（背景全透明）
    """
    from PIL import Image

    arr = _get_cached_array(nii_path)   # float32, (z,y,x)
    slice_2d, index = _extract_slice(arr, axis, index)

    h, w = slice_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 获取所有非零标签值并着色
    labels = np.unique(slice_2d[slice_2d != 0]).astype(int)
    for label in labels:
        if color_rgb is not None:
            color = color_rgb
        else:
            color = _LABEL_COLORS[(label - 1) % len(_LABEL_COLORS)]
        mask = slice_2d == label
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = min(255, max(0, alpha))

    pil_img = Image.fromarray(rgba, mode='RGBA')

    # 体素间距等比校正（与 CT 切片保持一致）
    if axis == 'y' and spacing_x > 0:
        scale = spacing_z / spacing_x
        if abs(scale - 1.0) > 0.02:
            nw, nh = pil_img.size
            new_nh = max(1, int(round(nh * scale)))
            pil_img = pil_img.resize((nw, new_nh), Image.NEAREST)
    elif axis == 'x' and spacing_y > 0:
        scale = spacing_z / spacing_y
        if abs(scale - 1.0) > 0.02:
            nw, nh = pil_img.size
            new_nh = max(1, int(round(nh * scale)))
            pil_img = pil_img.resize((nw, new_nh), Image.NEAREST)

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


def get_nii_metadata(nii_path: str) -> dict:
    """
    解析 NIfTI 文件元数据，返回与 parse_mhd_metadata 相同格式的字典。
    """
    return parse_volume_metadata(nii_path)


def list_nii_labels(nii_path: str) -> list:
    """
    返回 NIfTI 掩码中出现的所有非零标签值列表（已排序）。
    可用于前端显示类别图例。
    """
    arr = _get_cached_array(nii_path)
    labels = [int(v) for v in np.unique(arr) if v != 0]
    return sorted(labels)
