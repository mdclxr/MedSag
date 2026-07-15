import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from medseg.config import get_cache_folder
from functools import wraps
from flask import jsonify
from flask_login import current_user
import os

def get_db_path():
    return os.path.join(get_cache_folder(), 'users.db')

def init_db():
    """初始化数据库，创建用户表和任务分配表"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 创建用户表（包含角色字段）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                company TEXT,
                role TEXT DEFAULT 'annotator',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 检查是否需要添加 role 列（兼容旧数据库）
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'role' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'admin'")
        if 'is_active' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        if 'created_at' not in columns:
            # SQLite ALTER TABLE 不支持 DEFAULT CURRENT_TIMESTAMP，使用 Python 生成的当前时间常量
            from datetime import datetime
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(f"ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT '{now_str}'")
        
        # 创建任务分配表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                image_name TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(project_name, image_name)
            )
        ''')
        
        conn.commit()

def get_user_count():
    """获取用户总数"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]

def create_user(first_name, last_name, username, email, password, company, role=None):
    """创建用户，如果是第一个用户则自动设为管理员"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        
        # 如果没有指定角色，根据是否是第一个用户决定
        if role is None:
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            role = 'admin' if user_count == 0 else 'annotator'
        
        try:
            cursor.execute('''
                INSERT INTO users (first_name, last_name, username, email, password_hash, company, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (first_name, last_name, username, email, password_hash, company, role))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def update_user(user_id, updates):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            set_clause = ', '.join(f"{key} = ?" for key in updates)
            values = list(updates.values()) + [user_id]
            cursor.execute(f'''
                UPDATE users
                SET {set_clause}
                WHERE id = ?
            ''', values)
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False

def get_user_by_username(username):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, password_hash, first_name, last_name, email, company, role
            FROM users WHERE username = ?
        ''', (username,))
        return cursor.fetchone()

def get_user_by_email(email):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, password_hash, first_name, last_name, email, company, role
            FROM users WHERE email = ?
        ''', (email,))
        return cursor.fetchone()

def get_user_by_id(user_id):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, password_hash, first_name, last_name, email, company, role
            FROM users WHERE id = ?
        ''', (user_id,))
        return cursor.fetchone()

# ==================== 用户管理功能 ====================

def get_all_users():
    """获取所有用户列表"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, first_name, last_name, email, company, role, is_active, created_at
            FROM users ORDER BY created_at DESC
        ''')
        users = cursor.fetchall()
        return [
            {
                'id': u[0],
                'username': u[1],
                'first_name': u[2],
                'last_name': u[3],
                'email': u[4],
                'company': u[5],
                'role': u[6],
                'is_active': u[7],
                'created_at': u[8]
            }
            for u in users
        ]

def get_all_annotators():
    """获取所有标注员列表"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, first_name, last_name, email
            FROM users WHERE role = 'annotator' AND is_active = 1
        ''')
        annotators = cursor.fetchall()
        return [
            {
                'id': a[0],
                'username': a[1],
                'first_name': a[2],
                'last_name': a[3],
                'email': a[4],
                'display_name': f"{a[2]} {a[3]} ({a[1]})"
            }
            for a in annotators
        ]

def update_user_role(user_id, new_role):
    """更新用户角色"""
    if new_role not in ['admin', 'annotator']:
        return False
    return update_user(user_id, {'role': new_role})

def toggle_user_active(user_id):
    """切换用户激活状态"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_active FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            new_status = 0 if result[0] == 1 else 1
            cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
            conn.commit()
            return True
        return False

def delete_user(user_id):
    """删除用户"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # 先删除该用户的任务分配记录
        cursor.execute("DELETE FROM image_assignments WHERE user_id = ?", (user_id,))
        # 再删除用户
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0

# ==================== 任务分配功能 ====================

def assign_images_to_user(project_name, image_names, user_id, completed_images=None):
    """将图片分配给用户"""
    db_path = get_db_path()
    completed_images = set(completed_images) if completed_images else set()
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        assigned_count = 0
        for image_name in image_names:
            # SAFEGUARD: Check if image is already completed in assignments
            # This protects against any upstream filtering failures
            cursor.execute('SELECT status FROM image_assignments WHERE project_name=? AND image_name=?', (project_name, image_name))
            row = cursor.fetchone()
            if row and row[0] == 'completed':
                # Skip re-assigning this image to preserve its status and owner
                continue

            status = 'completed' if image_name in completed_images else 'pending'
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO image_assignments (project_name, image_name, user_id, status)
                    VALUES (?, ?, ?, ?)
                ''', (project_name, image_name, user_id, status))
                assigned_count += 1
            except sqlite3.IntegrityError:
                pass 
        conn.commit()
        return assigned_count

def auto_assign_images(project_name, image_names, annotator_ids, completed_images=None):
    """自动平均分配图片给多个标注员"""
    if not annotator_ids:
        return {}
    
    assignments = {}
    for i, image_name in enumerate(image_names):
        annotator_id = annotator_ids[i % len(annotator_ids)]
        if annotator_id not in assignments:
            assignments[annotator_id] = []
        assignments[annotator_id].append(image_name)
    
    # 执行分配
    for user_id, images in assignments.items():
        assign_images_to_user(project_name, images, user_id, completed_images)
    
    return assignments

def get_user_assigned_images(project_name, user_id):
    """获取分配给用户的图片列表"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT image_name, status, assigned_at, completed_at
            FROM image_assignments
            WHERE project_name = ? AND user_id = ?
        ''', (project_name, user_id))
        return [
            {
                'image_name': row[0],
                'status': row[1],
                'assigned_at': row[2],
                'completed_at': row[3]
            }
            for row in cursor.fetchall()
        ]

def get_project_assignments(project_name):
    """获取项目的所有分配情况"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ia.image_name, ia.user_id, u.username, u.first_name, u.last_name, ia.status
            FROM image_assignments ia
            JOIN users u ON ia.user_id = u.id
            WHERE ia.project_name = ?
        ''', (project_name,))
        return [
            {
                'image_name': row[0],
                'user_id': row[1],
                'username': row[2],
                'annotator_name': f"{row[3]} {row[4]}",
                'status': row[5]
            }
            for row in cursor.fetchall()
        ]

def get_user_projects(user_id):
    """获取用户被分配任务的项目列表"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT project_name FROM image_assignments WHERE user_id = ?
        ''', (user_id,))
        return [row[0] for row in cursor.fetchall()]

def update_image_status(project_name, image_name, status):
    """更新图片标注状态"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if status == 'completed':
            cursor.execute('''
                UPDATE image_assignments 
                SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE project_name = ? AND image_name = ?
            ''', (status, project_name, image_name))
        else:
            cursor.execute('''
                UPDATE image_assignments 
                SET status = ?
                WHERE project_name = ? AND image_name = ?
            ''', (status, project_name, image_name))
        conn.commit()
        return cursor.rowcount > 0

def get_project_progress(project_name):
    """获取项目标注进度统计"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM image_assignments
            WHERE project_name = ?
        ''', (project_name,))
        row = cursor.fetchone()
        return {
            'total': row[0] or 0,
            'completed': row[1] or 0,
            'in_progress': row[2] or 0,
            'pending': row[3] or 0
        }

def get_user_progress(project_name, user_id):
    """获取用户在某项目的标注进度"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
            FROM image_assignments
            WHERE project_name = ? AND user_id = ?
        ''', (project_name, user_id))
        row = cursor.fetchone()
        return {
            'total': row[0] or 0,
            'completed': row[1] or 0
        }

def unassign_image(project_name, image_name):
    """取消图片分配"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM image_assignments WHERE project_name = ? AND image_name = ?
        ''', (project_name, image_name))
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM image_assignments WHERE project_name = ? AND image_name = ?
        ''', (project_name, image_name))
        conn.commit()
        return cursor.rowcount > 0

def clear_completed_assignments(project_name, user_id):
    """清除用户在某项目中的已完成任务记录（重置进度）"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM image_assignments 
            WHERE project_name = ? AND user_id = ? AND status = 'completed'
        ''', (project_name, user_id))
        conn.commit()
        return cursor.rowcount > 0

def cleanup_project_assignments(project_name, valid_image_names):
    """清除项目中无效图片的分配记录"""
    if not valid_image_names:
        return 0
        
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 获取该项目当前所有分配的图片名
        cursor.execute('SELECT id, image_name FROM image_assignments WHERE project_name = ?', (project_name,))
        assignments = cursor.fetchall()
        
        ids_to_delete = []
        for aid, img_name in assignments:
            # 比较时不区分大小写
            if img_name not in valid_image_names and img_name.lower() not in valid_image_names:
                ids_to_delete.append((aid,))
        
        if ids_to_delete:
            cursor.executemany('DELETE FROM image_assignments WHERE id = ?', ids_to_delete)
            conn.commit()
            return len(ids_to_delete)
        return 0

# ==================== 权限装饰器 ====================

def admin_required(f):
    """管理员权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'message': '请先登录'}), 401
        if current_user.role != 'admin':
            return jsonify({'success': False, 'message': '需要管理员权限'}), 403
        return f(*args, **kwargs)
    return decorated_function

# ==================== User 类 ====================

class User(UserMixin):
    def __init__(self, user_id, username, first_name, last_name, email, company, role='annotator'):
        self.id = user_id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.company = company
        self.role = role

    @property
    def avatar(self):
        return f"{self.first_name[0]}.{self.last_name[0]}" if self.first_name and self.last_name else ""
    
    @property
    def is_admin(self):
        return self.role == 'admin'
    
    @property
    def is_annotator(self):
        return self.role == 'annotator'
    
    @property
    def display_name(self):
        return f"{self.first_name} {self.last_name}"