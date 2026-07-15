import os
import platform


def _pick_first_existing(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else ''


def get_cache_folder():
    """返回 MedSeg 数据缓存根目录，可通过环境变量 MEDSEG_DATA_DIR 自定义。"""
    custom_cache_dir = os.environ.get('MEDSEG_DATA_DIR', '/opt/medseg_data')
    os.makedirs(custom_cache_dir, exist_ok=True)
    return custom_cache_dir


def get_db_path():
    return os.path.join(get_cache_folder(), 'users.db')


PROJECTS_FOLDER = get_cache_folder()

# CT 数据文件扩展名
VALID_CT_EXTENSIONS = {'.mhd', '.raw'}
VALID_NIFTI_EXTENSIONS = {'.nii', '.nii.gz'}

# 项目类型（仅保留医学CT分割）
VALID_SETUP_TYPES = {
    "CT 肺结节检测",
}

# ── MedSAM2 配置 ──────────────────────────────────────────────────────────────
MEDSAM2_SERVICE_URL = os.environ.get('MEDSAM2_SERVICE_URL', 'http://127.0.0.1:7001')
MEDSAM2_SERVICE_TIMEOUT = int(os.environ.get('MEDSAM2_SERVICE_TIMEOUT', '15'))
MEDSAM2_REPO = os.environ.get('MEDSAM2_REPO', '/home/mdc/MedSAM2')
MEDSAM2_CHECKPOINT = os.environ.get(
    'MEDSAM2_CHECKPOINT',
    _pick_first_existing(
        os.path.join(MEDSAM2_REPO, 'checkpoints', 'MedSAM2_latest.pt'),
        os.path.join(MEDSAM2_REPO, 'checkpoints', 'MedSAM2_2411.pt'),
    )
)
MEDSAM2_CONFIG = os.environ.get(
    'MEDSAM2_CONFIG',
    _pick_first_existing(
        os.path.join(MEDSAM2_REPO, 'sam2', 'configs', 'sam2.1_hiera_t512.yaml'),
        os.path.join(MEDSAM2_REPO, 'efficient_track_anything', 'configs', 'efficienttam_s_512x512.yaml'),
    )
)
MEDSAM2_DEVICE = os.environ.get('MEDSAM2_DEVICE', 'cuda:0')

# ── MedSAM3 配置（预留，后续集成用）─────────────────────────────────────────
MEDSAM3_SERVICE_URL = os.environ.get('MEDSAM3_SERVICE_URL', 'http://127.0.0.1:7002')
MEDSAM3_SERVICE_TIMEOUT = int(os.environ.get('MEDSAM3_SERVICE_TIMEOUT', '15'))
MEDSAM3_REPO = os.environ.get('MEDSAM3_REPO', '/home/mdc/MedSAM3')
MEDSAM3_CHECKPOINT = os.environ.get('MEDSAM3_CHECKPOINT', '')
MEDSAM3_DEVICE = os.environ.get('MEDSAM3_DEVICE', 'cuda:0')

# 其他
WEIGHTS_FOLDER = os.path.join(get_cache_folder(), 'weights')
TMP_FOLDER = os.path.join(get_cache_folder(), 'tmp')
DATASETS_FOLDER = os.path.join(PROJECTS_FOLDER, 'datasets')


class Config:
    SECRET_KEY = os.environ.get('MEDSEG_SECRET_KEY', 'MEDSEG_SECRET')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024   # 500 MB 单次上传上限
    MAX_FORM_PARTS = 1000000
    PROJECTS_FOLDER = PROJECTS_FOLDER
    DATABASE_PATH = get_db_path()
    DATASETS_FOLDER = DATASETS_FOLDER
    LOGIN_DISABLED = False
    # MedSAM2
    MEDSAM2_SERVICE_URL = MEDSAM2_SERVICE_URL
    MEDSAM2_SERVICE_TIMEOUT = MEDSAM2_SERVICE_TIMEOUT
    MEDSAM2_REPO = MEDSAM2_REPO
    MEDSAM2_CHECKPOINT = MEDSAM2_CHECKPOINT
    MEDSAM2_CONFIG = MEDSAM2_CONFIG
    MEDSAM2_DEVICE = MEDSAM2_DEVICE
    # MedSAM3（预留）
    MEDSAM3_SERVICE_URL = MEDSAM3_SERVICE_URL
    MEDSAM3_SERVICE_TIMEOUT = MEDSAM3_SERVICE_TIMEOUT
    MEDSAM3_REPO = MEDSAM3_REPO
    MEDSAM3_CHECKPOINT = MEDSAM3_CHECKPOINT
    MEDSAM3_DEVICE = MEDSAM3_DEVICE
