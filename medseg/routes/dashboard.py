"""
MedSeg Dashboard 路由（裁剪版）
仅保留 CT 医学分割项目相关功能，去除图像/视频/训练/数据集等非医学部分。
"""
import logging
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import shutil
import zipfile
import tarfile
import sqlite3
from filelock import FileLock
import time
import psutil
import errno
import uuid

from medseg.config import (
    PROJECTS_FOLDER,
    VALID_CT_EXTENSIONS,
    VALID_SETUP_TYPES,
    get_cache_folder,
)
from medseg.models.project import Project
from medseg.routes.ct_viewer import initialize_ct_volume_mask
from medseg.models.user import (
    admin_required, get_all_users, get_all_annotators,
    update_user_role, toggle_user_active, delete_user,
    assign_images_to_user, auto_assign_images, get_user_assigned_images,
    get_project_assignments, get_user_projects, get_project_progress,
    get_user_progress, cleanup_project_assignments, clear_completed_assignments,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

bp = Blueprint('dashboard', __name__)

VALID_NIFTI_EXT = {'.nii', '.nii.gz'}


def _safe_storage_filename(filename):
    base_name = os.path.basename((filename or '').strip()).replace('\x00', '')
    if not base_name:
        return f"file_{uuid.uuid4().hex[:8]}"
    stem, extension = os.path.splitext(base_name)
    safe_stem = secure_filename(stem)
    safe_ext = extension.lower()
    if not safe_stem:
        safe_stem = f"file_{uuid.uuid4().hex[:8]}"
    return f"{safe_stem}{safe_ext}" if safe_ext else safe_stem


def generate_unique_project_name():
    base_name = "#MedSeg"
    counter = 0
    while True:
        project_name = base_name if counter == 0 else f"{base_name}_{counter}"
        if not os.path.exists(os.path.join(PROJECTS_FOLDER, project_name)):
            return project_name
        counter += 1


def ensure_unique_project_name(project_name):
    original_name = secure_filename(project_name)
    counter = 1
    new_name = original_name
    while os.path.exists(os.path.join(PROJECTS_FOLDER, new_name)):
        new_name = f"{original_name}_{counter}"
        counter += 1
    return new_name


# ──────────────────────────────────────────────────────────────────────────────
# 主页：CT 项目列表
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/')
@login_required
def index():
    projects = []
    if current_user.role == 'annotator':
        assigned_project_names = get_user_projects(current_user.id)
    else:
        assigned_project_names = None

    for project_name in os.listdir(PROJECTS_FOLDER):
        if project_name in ['temp_chunks', 'weights', 'datasets']:
            continue
        if assigned_project_names is not None and project_name not in assigned_project_names:
            continue

        project_path = os.path.join(PROJECTS_FOLDER, project_name)
        if not os.path.isdir(project_path):
            continue

        db_path = os.path.join(project_path, 'config.db')
        creation_date = None
        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT creation_date FROM Project_Configuration WHERE project_name = ?', (project_name,))
                    result = cursor.fetchone()
                    creation_date = result[0] if result else None
            except Exception as e:
                logger.error(f"Error fetching creation date for {project_name}: {e}")

        project = Project(project_name, '', '', project_path)
        setup_type = project.get_setup_type() or ''

        # 只展示 CT 项目
        if 'CT' not in setup_type:
            continue

        media_urls = []
        try:
            volumes = project.get_ct_volumes()
            for vol in volumes[:3]:
                preview_path = vol.get('preview_path')
                if preview_path and os.path.isfile(preview_path):
                    try:
                        rel = os.path.relpath(preview_path, PROJECTS_FOLDER).replace('\\', '/')
                        media_urls.append(f'/projects/{rel}')
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning(f"获取 CT 预览失败 {project_name}: {e}")

        if current_user.role == 'annotator':
            progress = get_user_progress(project_name, current_user.id)
        else:
            ct_count = project.get_ct_volume_count()
            progress = {
                'total': ct_count,
                'completed': min(ct_count, project.get_ct_reviewed_count()),
                'is_ct': True
            }

        projects.append({
            'name': project_name,
            'images': media_urls,
            'setup_type': setup_type,
            'creation_date': creation_date,
            'progress': progress
        })

    projects.sort(key=lambda p: p['creation_date'] or '', reverse=True)
    return render_template('index.html', projects=projects)


# ──────────────────────────────────────────────────────────────────────────────
# 文件上传（分块上传 + 组装）
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/upload_chunk', methods=['POST'])
@login_required
def upload_chunk():
    if 'chunk' not in request.files:
        return jsonify({'error': 'No chunk provided'}), 400

    chunk = request.files['chunk']
    upload_id = request.form.get('upload_id')
    file_id = request.form.get('file_id')
    chunk_index = int(request.form.get('chunk_index'))
    filename = _safe_storage_filename(request.form.get('filename'))

    if not all([upload_id, file_id, filename]):
        return jsonify({'error': 'Missing upload parameters'}), 400

    cache_dir = get_cache_folder()
    temp_base = os.path.join(cache_dir, 'temp_chunks')
    os.makedirs(temp_base, exist_ok=True)
    temp_dir = os.path.join(temp_base, upload_id, file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # 清理超过1小时的临时文件
    try:
        current_time = time.time()
        for temp_upload_id in os.listdir(temp_base):
            temp_upload_path = os.path.join(temp_base, temp_upload_id)
            if os.path.isdir(temp_upload_path):
                mtime = os.path.getmtime(temp_upload_path)
                if current_time - mtime > 3600:
                    shutil.rmtree(temp_upload_path, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Error cleaning up stale temp files: {e}")

    chunk_path = os.path.join(temp_dir, f'chunk_{chunk_index}')
    try:
        chunk.save(chunk_path)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving chunk {chunk_index} for {filename}: {e}")
        return jsonify({'error': f'Chunk save failed: {str(e)}'}), 500


@bp.route('/assemble_file', methods=['POST'])
@login_required
def assemble_file():
    import hashlib
    from shutil import copyfileobj

    upload_id = request.form.get('upload_id')
    file_id = request.form.get('file_id')
    total_chunks = int(request.form.get('total_chunks', 0))
    filename = _safe_storage_filename(request.form.get('filename'))
    expected_hash = request.form.get('file_hash', '')

    if not all([upload_id, file_id, filename, total_chunks]):
        return jsonify({'error': 'Missing assembly parameters'}), 400

    cache_dir = get_cache_folder()
    temp_base = os.path.join(cache_dir, 'temp_chunks')
    temp_dir = os.path.join(temp_base, upload_id, file_id)
    final_dir = os.path.join(temp_base, upload_id)
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, filename)
    lock_path = final_path + '.lock'

    try:
        with FileLock(lock_path):
            with open(final_path, 'wb') as f:
                for i in range(total_chunks):
                    chunk_path = os.path.join(temp_dir, f'chunk_{i}')
                    if not os.path.exists(chunk_path):
                        raise FileNotFoundError(f'Missing chunk {i}')
                    with open(chunk_path, 'rb') as chunk_file:
                        copyfileobj(chunk_file, f)
                    os.remove(chunk_path)

            if expected_hash:
                with open(final_path, 'rb') as f:
                    hasher = hashlib.md5()
                    while chunk_data := f.read(4096):
                        hasher.update(chunk_data)
                    assembled_hash = hasher.hexdigest()
                if assembled_hash != expected_hash:
                    os.remove(final_path)
                    return jsonify({'error': 'File corrupted during assembly (hash mismatch)'}), 400

            try:
                os.rmdir(temp_dir)
            except OSError as e:
                if e.errno != errno.ENOTEMPTY:
                    logger.warning(f"Cleanup warning for {temp_dir}: {str(e)}")

            return jsonify({'success': True, 'file_path': final_path})
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error assembling {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Assembly failed due to server error'}), 500


@bp.route('/check_upload_status', methods=['POST'])
@login_required
def check_upload_status():
    upload_id = request.form.get('upload_id')
    file_id = request.form.get('file_id')
    cache_dir = get_cache_folder()
    temp_base = os.path.join(cache_dir, 'temp_chunks')
    temp_dir = os.path.join(temp_base, upload_id, file_id)
    if not os.path.exists(temp_dir):
        return jsonify({'uploaded_chunks': 0})
    uploaded_chunks = len([f for f in os.listdir(temp_dir) if f.startswith('chunk_')])
    return jsonify({'uploaded_chunks': uploaded_chunks})


# ──────────────────────────────────────────────────────────────────────────────
# CT 项目创建
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/get_unique_project_name', methods=['GET'])
@login_required
def get_unique_project_name():
    try:
        project_name = generate_unique_project_name()
        return jsonify({'success': True, 'project_name': project_name})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@bp.route('/parse_annotations', methods=['POST'])
@login_required
def parse_annotations():
    return jsonify({
        'success': True,
        'summary': {
            'coco_files': [],
            'yolo_files': [],
            'class_mapping': {},
            'annotated_images': 0
        }
    })


@bp.route('/create_project', methods=['POST'])
@login_required
def create_project():
    try:
        project_name = request.form.get('project_name', '').strip()
        description = request.form.get('description', '')
        setup_type = request.form.get('setup_type', '').strip()
        class_names = request.form.get('class_names', '')
        upload_id = request.form.get('upload_id')

        if not setup_type or not upload_id:
            return jsonify({'error': 'Setup type and upload ID are required'}), 400
        if setup_type not in VALID_SETUP_TYPES:
            return jsonify({'error': f'不支持的项目类型: {setup_type}'}), 400

        class_list = [cls.strip() for cls in class_names.replace(';', ',').replace('.', ',').split(',') if cls.strip()]

        if not project_name:
            project_name = generate_unique_project_name()
        else:
            project_name = ensure_unique_project_name(project_name)

        project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
        ct_volumes_path = os.path.join(project_path, 'ct_volumes')
        previews_path = os.path.join(ct_volumes_path, 'previews')
        os.makedirs(ct_volumes_path, exist_ok=True)
        os.makedirs(previews_path, exist_ok=True)

        project = Project(project_name, description, setup_type, project_path)
        project.add_classes(class_list)

        cache_dir = get_cache_folder()
        temp_base = os.path.join(cache_dir, 'temp_chunks')
        temp_upload_dir = os.path.join(temp_base, upload_id)
        if not os.path.exists(temp_upload_dir):
            return jsonify({'error': 'No files found for upload ID'}), 400

        ct_img_map  = {}
        ct_mask_map = {}
        ct_csv_map  = {}

        for filename in os.listdir(temp_upload_dir):
            file_path = os.path.join(temp_upload_dir, filename)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            fname_lower = filename.lower()
            if fname_lower.endswith('.nii.gz'):
                ext = '.nii.gz'

            if ext == '.mhd':
                final_path = os.path.join(ct_volumes_path, _safe_storage_filename(filename))
                if not os.path.exists(final_path):
                    lock_path = final_path + '.lock'
                    with FileLock(lock_path):
                        shutil.copy2(file_path, final_path)
                    raw_name = os.path.splitext(filename)[0] + '.raw'
                    raw_src = os.path.join(temp_upload_dir, raw_name)
                    raw_dst = os.path.join(ct_volumes_path, os.path.splitext(_safe_storage_filename(filename))[0] + '.raw')
                    if os.path.isfile(raw_src) and not os.path.exists(raw_dst):
                        shutil.copy2(raw_src, raw_dst)
                stem = os.path.splitext(os.path.basename(final_path))[0]
                ct_img_map[stem] = os.path.abspath(final_path)

            elif ext == '.raw':
                continue  # 由 .mhd 处理时一并搬运

            elif ext in ('.nii', '.nii.gz'):
                final_path = os.path.join(ct_volumes_path, filename)
                if not os.path.exists(final_path):
                    shutil.copy2(file_path, final_path)
                final_path = os.path.abspath(final_path)

                if fname_lower.endswith('_mask.nii.gz') or fname_lower.endswith('_mask.nii'):
                    base = fname_lower.replace('_mask.nii.gz', '').replace('_mask.nii', '')
                    ct_mask_map[base] = final_path
                    ct_mask_map['__any__'] = final_path
                elif fname_lower.endswith('_img.nii.gz') or fname_lower.endswith('_img.nii'):
                    base = fname_lower.replace('_img.nii.gz', '').replace('_img.nii', '')
                    ct_img_map[base] = final_path
                else:
                    base = fname_lower.replace('.nii.gz', '').replace('.nii', '')
                    if base not in ct_img_map:
                        ct_img_map[base] = final_path

            elif ext == '.csv':
                stem = os.path.splitext(filename)[0].lower()
                ct_csv_map[stem] = file_path
                ct_csv_map['__any__'] = file_path

        if not ct_img_map:
            shutil.rmtree(project_path, ignore_errors=True)
            return jsonify({'error': '未找到有效的 CT 文件，请上传 .mhd+.raw 或 *_img.nii.gz 文件'}), 400

        from medseg.utils.ct_utils import generate_preview_png
        for stem, img_path in ct_img_map.items():
            mask_path = ct_mask_map.get(stem) or ct_mask_map.get('__any__')
            csv_path_found = ct_csv_map.get(stem) or ct_csv_map.get('__any__')
            safe_stem = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
            preview_path = os.path.join(previews_path, f"{safe_stem}_preview.png")
            generate_preview_png(img_path, preview_path)
            volume_id = project.add_ct_volume(img_path, preview_path,
                                               nii_path=mask_path,
                                               csv_path=csv_path_found)
            if volume_id and csv_path_found:
                project.add_ct_annotations_from_csv(volume_id, csv_path_found, series_id=stem)
            if volume_id and not mask_path:
                auto_action = 'create_from_csv' if csv_path_found else 'create_blank'
                try:
                    auto_mask = initialize_ct_volume_mask(project, volume_id, action=auto_action)
                    mask_path = auto_mask['nii_path']
                except Exception as auto_mask_error:
                    logger.warning(f"CT 自动初始化掩码失败 {img_path}: {auto_mask_error}")
            logger.info(f"CT 体数据已注册: {img_path} mask={mask_path}")

        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        return jsonify({'success': True, 'project_name': project_name})
    except Exception as e:
        logger.error(f"Error in create_project: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@bp.route('/import_images', methods=['POST'])
@login_required
def import_images():
    """向已有 CT 项目追加导入体数据"""
    try:
        project_name = request.form.get('project_name')
        upload_id = request.form.get('upload_id')
        if not project_name or not upload_id:
            return jsonify({'error': 'Project name and upload ID are required'}), 400

        project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
        if not os.path.exists(project_path):
            return jsonify({'error': 'Project does not exist'}), 404

        project = Project(project_name, '', '', project_path)
        ct_volumes_path = os.path.join(project_path, 'ct_volumes')
        previews_path = os.path.join(ct_volumes_path, 'previews')
        os.makedirs(ct_volumes_path, exist_ok=True)
        os.makedirs(previews_path, exist_ok=True)

        cache_dir = get_cache_folder()
        temp_base = os.path.join(cache_dir, 'temp_chunks')
        temp_upload_dir = os.path.join(temp_base, upload_id)
        if not os.path.exists(temp_upload_dir):
            return jsonify({'error': 'No files found for upload ID'}), 400

        ct_img_map  = {}
        ct_mask_map = {}
        ct_csv_map  = {}

        for filename in os.listdir(temp_upload_dir):
            file_path = os.path.join(temp_upload_dir, filename)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            fname_lower = filename.lower()
            if fname_lower.endswith('.nii.gz'):
                ext = '.nii.gz'

            if ext == '.mhd':
                final_path = os.path.join(ct_volumes_path, _safe_storage_filename(filename))
                if not os.path.exists(final_path):
                    lock_path = final_path + '.lock'
                    with FileLock(lock_path):
                        shutil.copy2(file_path, final_path)
                    raw_name = os.path.splitext(filename)[0] + '.raw'
                    raw_src = os.path.join(temp_upload_dir, raw_name)
                    raw_dst = os.path.join(ct_volumes_path, os.path.splitext(_safe_storage_filename(filename))[0] + '.raw')
                    if os.path.isfile(raw_src) and not os.path.exists(raw_dst):
                        shutil.copy2(raw_src, raw_dst)
                stem = os.path.splitext(os.path.basename(final_path))[0].lower()
                ct_img_map[stem] = os.path.abspath(final_path)
            elif ext == '.raw':
                continue
            elif ext in ('.nii', '.nii.gz'):
                final_path = os.path.join(ct_volumes_path, filename)
                if not os.path.exists(final_path):
                    shutil.copy2(file_path, final_path)
                final_path = os.path.abspath(final_path)
                if fname_lower.endswith('_mask.nii.gz') or fname_lower.endswith('_mask.nii'):
                    base = fname_lower.replace('_mask.nii.gz', '').replace('_mask.nii', '')
                    ct_mask_map[base] = final_path
                    ct_mask_map['__any__'] = final_path
                elif fname_lower.endswith('_img.nii.gz') or fname_lower.endswith('_img.nii'):
                    base = fname_lower.replace('_img.nii.gz', '').replace('_img.nii', '')
                    ct_img_map[base] = final_path
                else:
                    base = fname_lower.replace('.nii.gz', '').replace('.nii', '')
                    if base not in ct_img_map:
                        ct_img_map[base] = final_path
            elif ext == '.csv':
                stem = os.path.splitext(filename)[0].lower()
                ct_csv_map[stem] = file_path
                ct_csv_map['__any__'] = file_path

        if not ct_img_map:
            return jsonify({'error': 'No new valid CT volumes found to import'}), 400

        from medseg.utils.ct_utils import generate_preview_png
        imported_count = 0
        for stem, img_path in ct_img_map.items():
            mask_path = ct_mask_map.get(stem) or ct_mask_map.get('__any__')
            csv_path_found = ct_csv_map.get(stem) or ct_csv_map.get('__any__')
            safe_stem = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
            preview_path = os.path.join(previews_path, f"{safe_stem}_preview.png")
            generate_preview_png(img_path, preview_path)
            volume_id = project.add_ct_volume(img_path, preview_path, nii_path=mask_path, csv_path=csv_path_found)
            if volume_id and csv_path_found:
                project.add_ct_annotations_from_csv(volume_id, csv_path_found, series_id=stem)
            if volume_id and not mask_path:
                auto_action = 'create_from_csv' if csv_path_found else 'create_blank'
                try:
                    auto_mask = initialize_ct_volume_mask(project, volume_id, action=auto_action)
                except Exception as auto_mask_error:
                    logger.warning(f"CT 导入后自动初始化掩码失败 {img_path}: {auto_mask_error}")
            if volume_id:
                imported_count += 1

        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        logger.info(f"Imported {imported_count} CT volumes into project {project_name}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in import_images: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 项目操作：删除、重命名
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/delete_project/<project_name>', methods=['POST'])
@login_required
def delete_project(project_name):
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if os.path.exists(project_path):
        try:
            shutil.rmtree(project_path)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': f'Deletion failed: {str(e)}'}), 500
    return jsonify({'error': 'Project not found'}), 404


@bp.route('/rename_project', methods=['POST'])
@login_required
def rename_project():
    old_name = request.form.get('old_name', '').strip()
    new_name = request.form.get('new_name', '').strip()
    if not old_name or not new_name:
        return jsonify({'error': '名称不能为空'}), 400
    old_path = os.path.join(PROJECTS_FOLDER, secure_filename(old_name))
    new_path = os.path.join(PROJECTS_FOLDER, secure_filename(new_name))
    if not os.path.exists(old_path):
        return jsonify({'error': '项目不存在'}), 404
    if os.path.exists(new_path):
        return jsonify({'error': '新名称已被占用'}), 400
    try:
        os.rename(old_path, new_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/cleanup_chunks', methods=['POST'])
@login_required
def cleanup_chunks():
    cache_dir = get_cache_folder()
    temp_base = os.path.join(cache_dir, 'temp_chunks')
    try:
        if os.path.exists(temp_base):
            shutil.rmtree(temp_base, ignore_errors=True)
        os.makedirs(temp_base, exist_ok=True)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 项目概览统计（CT 版）
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/get_project_overview/<project_name>', methods=['GET'])
@login_required
def get_project_overview(project_name):
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'error': 'Project not found'}), 404

    try:
        import numpy as np
        from scipy import ndimage
        from medseg.utils.ct_utils import _get_cached_array

        project = Project(project_name, '', '', project_path)
        ct_classes = project.get_classes() or ['nodule']
        volumes = project.get_ct_volumes()
        total_volumes = len(volumes)
        reviewed_ids = project.get_ct_reviewed_volume_ids()
        reviewed_count = len(reviewed_ids)
        unreviewed_count = max(0, total_volumes - reviewed_count)

        with_csv = 0
        with_mask = 0
        total_findings = 0
        class_distribution = {}
        findings_per_volume = []
        volume_table = []

        def label_name(classes, value: int) -> str:
            idx = int(value) - 1
            if 0 <= idx < len(classes):
                return str(classes[idx]).strip() or f'class_{value}'
            return f'class_{value}'

        for volume in volumes:
            volume_id = int(volume['volume_id'])
            volume_name = str(volume.get('name') or f'volume_{volume_id}')
            csv_annos = project.get_ct_annotations_for_volume(volume_id)
            has_csv = len(csv_annos) > 0
            has_mask = bool(volume.get('nii_path') and os.path.isfile(volume.get('nii_path')))
            reviewed = volume_id in reviewed_ids

            findings = []
            if has_mask:
                try:
                    arr = _get_cached_array(volume['nii_path'])
                    positive = arr > 0
                    if np.any(positive):
                        labeled, count = ndimage.label(positive, structure=np.ones((3, 3, 3), dtype=bool))
                        objects = ndimage.find_objects(labeled)
                        per_class = {}
                        for li in range(1, count + 1):
                            slc = objects[li - 1]
                            if not slc:
                                continue
                            comp = labeled[slc] == li
                            raw_vals = arr[slc][comp]
                            raw_vals = raw_vals[raw_vals > 0]
                            if raw_vals.size == 0:
                                continue
                            dom_val = int(np.bincount(raw_vals.astype(int)).argmax())
                            dom_name = label_name(ct_classes, dom_val)
                            findings.append({'label_name': dom_name, 'voxel_count': int(comp.sum())})
                            per_class[dom_name] = per_class.get(dom_name, 0) + 1
                        for cn, cc in per_class.items():
                            class_distribution[cn] = class_distribution.get(cn, 0) + cc
                        with_mask += 1
                except Exception:
                    pass

            if not findings and has_csv:
                for anno in csv_annos:
                    lbl = str(anno.get('label') or 'nodule').strip() or 'nodule'
                    findings.append({'label_name': lbl, 'voxel_count': 0})
                    class_distribution[lbl] = class_distribution.get(lbl, 0) + 1

            if has_csv:
                with_csv += 1
            finding_count = len(findings)
            total_findings += finding_count
            findings_per_volume.append(finding_count)
            volume_table.append({
                'name': volume_name,
                'reviewed': reviewed,
                'has_mask': has_mask,
                'has_csv': has_csv,
                'finding_count': finding_count,
            })

        data = {
            'is_ct': True,
            'title_label': '总 CT 数',
            'total_images': total_volumes,
            'annotated_images': reviewed_count,
            'non_annotated_images': unreviewed_count,
            'completion_chart': {
                'labels': ['已确认保存', '未确认保存'],
                'values': [reviewed_count, unreviewed_count],
                'title': 'CT 标注完成状态',
            },
            'class_distribution': class_distribution,
            'annotations_per_image': findings_per_volume,
            'overview_metrics': [
                {'label': 'CT 总数', 'value': total_volumes},
                {'label': '已确认保存', 'value': reviewed_count},
                {'label': '含掩码 CT', 'value': with_mask},
                {'label': '含 CSV 标注 CT', 'value': with_csv},
                {'label': '检出病灶总数', 'value': total_findings},
            ],
            'ct_charts': {
                'class_title': '病灶类别统计（按 3D 目标数）',
                'class_x_title': '类别',
                'class_y_title': '3D 病灶数量',
                'hist_title': '每个 CT 的病灶数量分布',
                'hist_x_title': '单个 CT 中的病灶数量',
                'hist_y_title': 'CT 数量',
            },
            'volume_table': volume_table[:50],
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching overview for {project_name}: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 类别管理
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/get_project_classes/<project_name>', methods=['GET'])
@login_required
def get_project_classes(project_name):
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'error': 'Project not found'}), 404
    try:
        project = Project(project_name, '', '', project_path)
        classes = project.get_classes()
        class_info = [{'name': cls, 'annotation_count': project.get_class_annotation_count(cls)['total']} for cls in classes]
        return jsonify({'success': True, 'classes': class_info})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@bp.route('/add_project_class/<project_name>', methods=['POST'])
@login_required
def add_project_class(project_name):
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'error': 'Project not found'}), 404
    class_name = request.form.get('class_name', '').strip()
    if not class_name:
        return jsonify({'success': False, 'error': 'Class name is required'}), 400
    try:
        project = Project(project_name, '', '', project_path)
        if class_name in project.get_classes():
            return jsonify({'success': False, 'error': 'Class already exists'}), 400
        project.add_classes([class_name])
        return jsonify({'success': True, 'message': f"Class '{class_name}' added successfully"})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@bp.route('/delete_project_class/<project_name>', methods=['POST'])
@login_required
def delete_project_class(project_name):
    project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
    if not os.path.exists(project_path):
        return jsonify({'success': False, 'error': 'Project not found'}), 404
    class_name = request.form.get('class_name', '').strip()
    if not class_name:
        return jsonify({'success': False, 'error': 'Class name is required'}), 400
    try:
        project = Project(project_name, '', '', project_path)
        result = project.delete_class(class_name)
        if result['success']:
            return jsonify({'success': True, 'message': f"Class '{class_name}' deleted"})
        return jsonify({'success': False, 'error': 'Class not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 用户管理 API（仅管理员）
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/get_all_users', methods=['GET'])
@login_required
@admin_required
def api_get_all_users():
    try:
        users = get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get_annotators', methods=['GET'])
@login_required
@admin_required
def api_get_annotators():
    try:
        project_name = request.args.get('project_name')
        annotators = get_all_annotators()
        if project_name:
            project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
            if os.path.exists(project_path):
                project = Project(project_name, '', '', project_path)
                valid_volume_names = {str(v.get('name') or '') for v in project.get_ct_volumes()}
                cleaned = cleanup_project_assignments(project_name, valid_volume_names)
                if cleaned > 0:
                    logger.info(f"Auto-cleaned {cleaned} orphaned assignments for project {project_name}")
            for annotator in annotators:
                annotator['stats'] = get_user_progress(project_name, annotator['id'])
        return jsonify({'success': True, 'annotators': annotators})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/update_user_role', methods=['POST'])
@login_required
@admin_required
def api_update_user_role():
    try:
        user_id = request.form.get('user_id')
        new_role = request.form.get('role')
        if not user_id or not new_role:
            return jsonify({'success': False, 'error': '缺少参数'}), 400
        if int(user_id) == current_user.id:
            return jsonify({'success': False, 'error': '不能修改自己的角色'}), 400
        success = update_user_role(int(user_id), new_role)
        return jsonify({'success': success, 'message': '角色已更新' if success else '更新失败'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/toggle_user_active', methods=['POST'])
@login_required
@admin_required
def api_toggle_user_active():
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': '缺少参数'}), 400
        if int(user_id) == current_user.id:
            return jsonify({'success': False, 'error': '不能禁用自己'}), 400
        success = toggle_user_active(int(user_id))
        return jsonify({'success': success, 'message': '用户状态已更新' if success else '更新失败'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/delete_user', methods=['POST'])
@login_required
@admin_required
def api_delete_user():
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': '缺少参数'}), 400
        if int(user_id) == current_user.id:
            return jsonify({'success': False, 'error': '不能删除自己'}), 400
        success = delete_user(int(user_id))
        return jsonify({'success': success, 'message': '用户已删除' if success else '删除失败'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# 任务分配 API（CT 版）
# ──────────────────────────────────────────────────────────────────────────────

@bp.route('/assign_images', methods=['POST'])
@login_required
@admin_required
def api_assign_images():
    try:
        project_name = request.form.get('project_name')
        user_id = request.form.get('user_id')
        annotator_ids = request.form.getlist('annotator_ids[]')
        if user_id and not annotator_ids:
            annotator_ids = [user_id]
        image_names = request.form.getlist('image_names[]')
        if not project_name or not annotator_ids or not image_names:
            return jsonify({'success': False, 'error': '缺少参数'}), 400
        valid_annotator_ids = [int(aid) for aid in annotator_ids if aid.isdigit()]
        if not valid_annotator_ids:
            return jsonify({'success': False, 'error': '无效的标注员ID'}), 400

        if request.form.get('reset_history') == 'true':
            for uid in valid_annotator_ids:
                clear_completed_assignments(project_name, uid)

        import random
        random.shuffle(image_names)
        n = len(valid_annotator_ids)
        base = len(image_names) // n
        rem = len(image_names) % n
        idx = 0
        total_assigned = 0
        for i, uid in enumerate(valid_annotator_ids):
            cnt = base + (1 if i < rem else 0)
            if cnt > 0:
                assign_images_to_user(project_name, image_names[idx:idx+cnt], uid)
                idx += cnt
                total_assigned += cnt
        return jsonify({'success': True, 'assigned_count': total_assigned,
                        'message': f'已分配 {total_assigned} 个 CT 给 {n} 位用户'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/auto_assign_images', methods=['POST'])
@login_required
@admin_required
def api_auto_assign_images():
    try:
        project_name = request.form.get('project_name')
        annotator_ids = request.form.getlist('annotator_ids[]')
        if not project_name or not annotator_ids:
            return jsonify({'success': False, 'error': '缺少参数'}), 400

        annotator_id_list = [int(id) for id in annotator_ids]
        if request.form.get('reset_history') == 'true':
            for uid in annotator_id_list:
                clear_completed_assignments(project_name, uid)

        project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
        if not os.path.exists(project_path):
            return jsonify({'success': False, 'error': '项目不存在'}), 404

        project = Project(project_name, '', '', project_path)
        image_names = [str(v.get('name') or '') for v in project.get_ct_volumes() if str(v.get('name') or '').strip()]
        if not image_names:
            return jsonify({'success': False, 'error': '项目中没有 CT 体数据'}), 400

        assignments = auto_assign_images(project_name, image_names, annotator_id_list)
        result = {uid: len(imgs) for uid, imgs in assignments.items()}
        return jsonify({'success': True, 'assignments': result,
                        'message': f'已自动分配 {len(image_names)} 个 CT 给 {len(annotator_ids)} 位用户'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get_project_assignments/<project_name>', methods=['GET'])
@login_required
def api_get_project_assignments(project_name):
    try:
        if current_user.role == 'admin':
            assignments = get_project_assignments(project_name)
        else:
            assignments = get_user_assigned_images(project_name, current_user.id)
        return jsonify({'success': True, 'assignments': assignments})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get_project_progress/<project_name>', methods=['GET'])
@login_required
def api_get_project_progress(project_name):
    try:
        project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
        if not os.path.exists(project_path):
            return jsonify({'success': False, 'error': '项目不存在'}), 404
        project = Project(project_name, '', '', project_path)
        total = project.get_ct_volume_count()
        completed = project.get_ct_reviewed_count()
        assignment_progress = get_project_progress(project_name)
        return jsonify({'success': True, 'progress': {
            'total_images': total,
            'completed_images': completed,
            'assigned_count': assignment_progress['total'],
            'unassigned_count': max(0, total - completed - assignment_progress['total']),
            'entity_type': 'ct',
            'is_ct': True,
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get_project_images/<project_name>', methods=['GET'])
@login_required
def api_get_project_images(project_name):
    try:
        project_path = os.path.join(PROJECTS_FOLDER, secure_filename(project_name))
        if not os.path.exists(project_path):
            return jsonify({'success': False, 'error': '项目不存在'}), 404
        project = Project(project_name, '', '', project_path)
        assignments = get_project_assignments(project_name)
        assignment_map = {a['image_name']: a for a in assignments}
        volumes = []
        for volume in project.get_ct_volumes():
            name = str(volume.get('name') or '').strip()
            if not name:
                continue
            assignment = assignment_map.get(name)
            preview_path = volume.get('preview_path')
            preview_url = None
            if preview_path and os.path.isfile(preview_path):
                try:
                    rel = os.path.relpath(preview_path, PROJECTS_FOLDER).replace('\\', '/')
                    preview_url = f'/projects/{rel}'
                except ValueError:
                    pass
            volumes.append({
                'name': name,
                'path': preview_url or '',
                'assigned_to': assignment['annotator_name'] if assignment else None,
                'status': assignment['status'] if assignment else 'unassigned',
                'is_completed': (assignment['status'] == 'completed') if assignment else False,
                'is_ct': True,
                'volume_id': volume['volume_id'],
                'slice_count': volume.get('shape_z') or 0,
            })
        return jsonify({'success': True, 'images': volumes, 'is_ct': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get_current_user_info', methods=['GET'])
@login_required
def api_get_current_user_info():
    return jsonify({'success': True, 'user': {
        'id': current_user.id,
        'username': current_user.username,
        'first_name': current_user.first_name,
        'last_name': current_user.last_name,
        'email': current_user.email,
        'role': current_user.role,
        'is_admin': current_user.is_admin
    }})
