import sqlite3
import os
import shutil
from PIL import Image
import json
import math
import logging
from medseg.utils import CocoAnnotationParser, YoloAnnotationParser, NameMatcher, is_valid_image

# Configure logging with less verbose output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class Project:
    def __init__(self, name, description, setup_type, project_path):
        self.name = name
        self.description = description
        self.setup_type = setup_type
        self.db_path = os.path.join(project_path, 'config.db')
        self.project_path = project_path
        self._initialize_db()
        self.setup_type = self.get_setup_type()
        self.videos_path = os.path.join(project_path, 'videos')
        
    def _initialize_db(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Project_Configuration (
                    project_name TEXT PRIMARY KEY,
                    description TEXT,
                    setup_type TEXT NOT NULL,
                    creation_date DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Classes (
                    class_name TEXT PRIMARY KEY
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    absolute_path TEXT UNIQUE,
                    width INTEGER,
                    height INTEGER,
                    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check for missing last_modified column in existing Images table
            cursor.execute("PRAGMA table_info(Images)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'last_modified' not in columns:
                try:
                    cursor.execute("ALTER TABLE Images ADD COLUMN last_modified DATETIME DEFAULT CURRENT_TIMESTAMP")
                except Exception as e:
                    logger.warning(f"Failed to add last_modified column to Images: {e}")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER,
                    user_id INTEGER,
                    type TEXT NOT NULL,
                    class_name TEXT,
                    x REAL,
                    y REAL,
                    width REAL,
                    height REAL,
                    rotation REAL DEFAULT 0,
                    segmentation TEXT,
                    FOREIGN KEY (image_id) REFERENCES Images(image_id),
                    FOREIGN KEY (class_name) REFERENCES Classes(class_name)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Preannotations (
                    preannotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER,
                    type TEXT NOT NULL,
                    class_name TEXT,
                    x REAL,
                    y REAL,
                    width REAL,
                    height REAL,
                    rotation REAL DEFAULT 0,
                    segmentation TEXT,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES Images(image_id),
                    FOREIGN KEY (class_name) REFERENCES Classes(class_name)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Videos (
                    video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    absolute_path TEXT UNIQUE,
                    name TEXT,
                    duration REAL,
                    fps REAL,
                    frame_count INTEGER,
                    width INTEGER,
                    height INTEGER
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Frames (
                    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    image_id INTEGER,
                    frame_number INTEGER,
                    subsampled BOOLEAN DEFAULT 0,
                    timestamp REAL,
                    FOREIGN KEY (video_id) REFERENCES Videos(video_id),
                    FOREIGN KEY (image_id) REFERENCES Images(image_id)
                )
            ''')
            # Create or update ReviewedImages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ReviewedImages (
                    image_id INTEGER PRIMARY KEY,
                    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER
                )
            ''')
            # Check if user_id column exists, and add it if not
            cursor.execute("PRAGMA table_info(ReviewedImages)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'user_id' not in columns:
                cursor.execute('''
                    ALTER TABLE ReviewedImages ADD COLUMN user_id INTEGER
                ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_absolute_path ON Images(absolute_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_annotations_image_id ON Annotations(image_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_preannotations_image_id ON Preannotations(image_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_video_id ON Frames(video_id)')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS CTVolumes (
                    volume_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    absolute_path TEXT UNIQUE,
                    name         TEXT,
                    shape_x      INTEGER,
                    shape_y      INTEGER,
                    shape_z      INTEGER,
                    spacing_x    REAL,
                    spacing_y    REAL,
                    spacing_z    REAL,
                    preview_path TEXT,
                    nii_path     TEXT,
                    csv_path     TEXT,
                    added_at     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 迁移旧表：补充新列
            cursor.execute("PRAGMA table_info(CTVolumes)")
            ct_cols = [c[1] for c in cursor.fetchall()]
            if 'nii_path' not in ct_cols:
                cursor.execute('ALTER TABLE CTVolumes ADD COLUMN nii_path TEXT')
            if 'csv_path' not in ct_cols:
                cursor.execute('ALTER TABLE CTVolumes ADD COLUMN csv_path TEXT')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctvolumes_path ON CTVolumes(absolute_path)')
            # CT 标注表：存储从 CSV 读取的结节世界坐标
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS CTAnnotations (
                    anno_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    volume_id    INTEGER NOT NULL,
                    label        TEXT DEFAULT 'nodule',
                    coord_x      REAL NOT NULL,
                    coord_y      REAL NOT NULL,
                    coord_z      REAL NOT NULL,
                    diameter_mm  REAL DEFAULT 0,
                    source       TEXT DEFAULT 'csv',
                    FOREIGN KEY (volume_id) REFERENCES CTVolumes(volume_id)
                )
            ''')
            cursor.execute("PRAGMA table_info(CTAnnotations)")
            anno_cols = [c[1] for c in cursor.fetchall()]
            if 'location' not in anno_cols:
                cursor.execute('ALTER TABLE CTAnnotations ADD COLUMN location TEXT')
            if 'texture' not in anno_cols:
                cursor.execute('ALTER TABLE CTAnnotations ADD COLUMN texture TEXT')
            if 'risk_level' not in anno_cols:
                cursor.execute('ALTER TABLE CTAnnotations ADD COLUMN risk_level TEXT')
            if 'signs' not in anno_cols:
                cursor.execute('ALTER TABLE CTAnnotations ADD COLUMN signs TEXT')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctanno_vol ON CTAnnotations(volume_id)')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS CTReviewedVolumes (
                    volume_id INTEGER PRIMARY KEY,
                    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER
                )
            ''')
            cursor.execute('''
                INSERT OR IGNORE INTO Project_Configuration (project_name, description, setup_type)
                VALUES (?, ?, ?)
            ''', (self.name, self.description, self.setup_type))
            conn.commit()

    def add_classes(self, class_list):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for cls in class_list:
                cursor.execute('INSERT OR IGNORE INTO Classes (class_name) VALUES (?)', (cls,))
            conn.commit()
            logger.info(f"Added {len(class_list)} classes to project {self.name}")

    def add_image(self, absolute_path):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                with Image.open(absolute_path) as img:
                    img.verify()  # Verify image integrity
                    img = Image.open(absolute_path)  # Reopen after verify
                    width, height = img.size
            except Exception as e:
                logger.error(f"Skipping corrupted image {absolute_path}: {str(e)}")
                return None
            
            cursor.execute('''
                INSERT OR IGNORE INTO Images (absolute_path, width, height)
                VALUES (?, ?, ?)
            ''', (absolute_path, width, height))
            conn.commit()
            cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (absolute_path,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Added image to database: {absolute_path}, image_id: {result[0]}")
                return result[0]
            else:
                logger.error(f"Failed to add image to database: {absolute_path}")
                return None

    def get_images(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_id, absolute_path, width, height, last_modified FROM Images')
            images = cursor.fetchall()
            logger.info(f"Retrieved {len(images)} images from database for project {self.name}")
            return images

    def get_images_dates(self):
        """Get a dictionary of image paths to their last modification time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT absolute_path, last_modified FROM Images')
            # If last_modified is NULL, we might default to something else, but here we just return what is DB.
            # Convert row[0] (path) to full URL format used in keys? No, path is absolute path.
            # The caller handles valid keys.
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_images_with_status(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT i.absolute_path, 
                       EXISTS (
                           SELECT 1 FROM Annotations a WHERE a.image_id = i.image_id
                       ) as is_annotated
                FROM Images i
            ''')
            return cursor.fetchall()

    def get_classes(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT class_name FROM Classes')
            classes = [row[0] for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(classes)} classes for project {self.name}: {classes}")
            return classes

    def delete_class(self, class_name):
        """从项目中删除指定类别，同时删除使用该类别的所有标注和预标注"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 统计将被删除的标注数量
            cursor.execute('SELECT COUNT(*) FROM Annotations WHERE class_name = ?', (class_name,))
            annotations_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM Preannotations WHERE class_name = ?', (class_name,))
            preannotations_count = cursor.fetchone()[0]
            
            # 删除相关标注
            cursor.execute('DELETE FROM Annotations WHERE class_name = ?', (class_name,))
            cursor.execute('DELETE FROM Preannotations WHERE class_name = ?', (class_name,))
            
            # 删除类别
            cursor.execute('DELETE FROM Classes WHERE class_name = ?', (class_name,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            logger.info(f"Deleted class '{class_name}' from project {self.name}, "
                       f"removed {annotations_count} annotations and {preannotations_count} preannotations")
            return {
                'success': deleted,
                'annotations_deleted': annotations_count,
                'preannotations_deleted': preannotations_count
            }

    def get_class_annotation_count(self, class_name):
        """获取指定类别的标注数量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM Annotations WHERE class_name = ?', (class_name,))
            annotations_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM Preannotations WHERE class_name = ?', (class_name,))
            preannotations_count = cursor.fetchone()[0]
            return {
                'annotations': annotations_count,
                'preannotations': preannotations_count,
                'total': annotations_count + preannotations_count
            }

    def get_setup_type(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT setup_type FROM Project_Configuration WHERE project_name = ?', (self.name,))
            result = cursor.fetchone()
            if result:
                return result[0]
            return None

    def add_images(self, absolute_paths):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            image_data = []
            for path in absolute_paths:
                try:
                    with Image.open(path) as img:
                        img.verify()  # Verify image integrity
                        img = Image.open(path)  # Reopen after verify
                        width, height = img.size
                    image_data.append((path, width, height))
                except Exception as e:
                    logger.error(f"Skipping corrupted image {path}: {str(e)}")
                    continue
            if image_data:
                cursor.executemany('''
                    INSERT OR IGNORE INTO Images (absolute_path, width, height)
                    VALUES (?, ?, ?)
                ''', image_data)
                conn.commit()
                logger.info(f"Added {len(image_data)} images to database for project {self.name}: {[path for path, _, _ in image_data]}")
            else:
                logger.warning(f"No valid images to add for project {self.name}")

    def add_video(self, absolute_path):
        try:
            import cv2
            cap = cv2.VideoCapture(absolute_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video {absolute_path}")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception as error:
            logger.error(f"Error getting video metadata for {absolute_path}: {str(error)}")
            return None

        name = os.path.basename(absolute_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO Videos (absolute_path, name, duration, fps, frame_count, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (absolute_path, name, duration, fps, frame_count, width, height))
            conn.commit()
            cursor.execute('SELECT video_id FROM Videos WHERE absolute_path = ?', (absolute_path,))
            result = cursor.fetchone()
            if result:
                return result[0]
            return None

    def add_selected_frames(self, video_id, target_fps=5):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT absolute_path, fps, frame_count, width, height FROM Videos WHERE video_id = ?', (video_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Video ID {video_id} not found")
                return 0
            absolute_path, fps, total_frames, width, height = result

        sampling = max(1, math.ceil(fps / target_fps)) if target_fps < fps else 1
        image_data = []
        frame_numbers = []
        timestamps = []
        for frame_num in range(total_frames):
            timestamp = frame_num / fps if fps and fps > 0 else 0
            virtual_path = f"{absolute_path}#{frame_num}"
            image_data.append((virtual_path, width, height))
            frame_numbers.append(frame_num)
            timestamps.append(timestamp)

        added = 0
        if image_data:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR IGNORE INTO Images (absolute_path, width, height)
                    VALUES (?, ?, ?)
                ''', image_data)
                conn.commit()

                frame_data = []
                for index, virtual_path in enumerate([item[0] for item in image_data]):
                    cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (virtual_path,))
                    image_id = cursor.fetchone()[0]
                    cursor.execute('SELECT frame_id FROM Frames WHERE video_id = ? AND frame_number = ?', (video_id, frame_numbers[index]))
                    if not cursor.fetchone():
                        is_subsampled = (frame_numbers[index] % sampling == 0)
                        frame_data.append((video_id, image_id, frame_numbers[index], is_subsampled, timestamps[index]))

                if frame_data:
                    cursor.executemany('''
                        INSERT INTO Frames (video_id, image_id, frame_number, subsampled, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', frame_data)
                    added = len(frame_data)
        return added

    def get_videos(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT video_id, absolute_path, name, duration, fps, frame_count FROM Videos')
            return cursor.fetchall()

    def get_video_path(self, video_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT absolute_path FROM Videos WHERE video_id = ?', (video_id,))
            result = cursor.fetchone()
            if result:
                return result[0]
            raise ValueError(f"Video ID {video_id} not found")

    def get_frames_for_video(self, video_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.frame_id, f.frame_number, f.subsampled
                FROM Frames f
                WHERE f.video_id = ?
                ORDER BY f.frame_number
            ''', (video_id,))
            return cursor.fetchall()

    def save_annotations(self, image_path, annotations, user_id=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (image_path,))
            image_id = cursor.fetchone()
            if not image_id:
                logger.error(f"Image {image_path} not found in database for project {self.name}")
                return
            image_id = image_id[0]

            # Delete existing annotations for this image
            cursor.execute('DELETE FROM Annotations WHERE image_id = ?', (image_id,))
            logger.info(f"Deleted existing annotations for {image_path} in project {self.name}")

            # Deduplicate annotations
            unique_annotations = []
            seen = set()
            for anno in annotations:
                anno_type = anno.get('type', 'rect')
                if self.setup_type == "Segmentation" and anno.get('segmentation'):
                    anno_type = 'polygon'
                elif self.setup_type == "Oriented Bounding Box":
                    anno_type = 'obbox'

                key = (anno_type, anno.get('category_name') or anno.get('label'))
                if anno.get('bbox'):
                    bbox = tuple(round(float(coord), 4) for coord in anno['bbox'])
                    rotation = round(float(anno.get('rotation', 0)), 4)
                    key += bbox + (rotation,)
                elif anno.get('segmentation'):
                    seg = anno['segmentation']
                    if isinstance(seg, list) and seg:
                        seg = seg[0] if isinstance(seg[0], list) else seg
                        sorted_seg = tuple(sorted(tuple(round(float(coord), 4) for coord in seg)))
                        key += sorted_seg

                if key not in seen:
                    seen.add(key)
                    unique_annotations.append(anno)

            # Save unique annotations
            for anno in unique_annotations:
                anno_type = anno.get('type', 'rect')
                if self.setup_type == "Segmentation" and anno.get('segmentation'):
                    anno_type = 'polygon'
                elif self.setup_type == "Oriented Bounding Box":
                    anno_type = 'obbox'

                x = y = width = height = rotation = segmentation = None
                if self.setup_type in ("Bounding Box", "Oriented Bounding Box"):
                    if anno.get('bbox'):
                        try:
                            x, y, width, height = map(float, anno['bbox'])
                            if width <= 0 or height <= 0:
                                logger.warning(f"Invalid bbox dimensions for {anno.get('category_name')} in {image_path}: width={width}, height={height}")
                                continue
                            if self.setup_type == "Oriented Bounding Box":
                                rotation = float(anno.get('rotation', 0))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid bbox format for {anno.get('category_name')} in {image_path}: {anno.get('bbox')}, error: {e}")
                            continue
                    else:
                        logger.warning(f"No bbox provided for {anno.get('category_name')} in {image_path}: {anno}")
                        continue
                elif self.setup_type == "Segmentation" and anno.get('segmentation'):
                    seg = anno['segmentation']
                    if isinstance(seg, list) and seg:
                        seg = seg[0] if isinstance(seg[0], list) else seg
                        segmentation = json.dumps(seg)
                    else:
                        logger.warning(f"Skipping invalid segmentation for {anno.get('category_name')} in {image_path}")
                        continue

                if (self.setup_type in ("Bounding Box", "Oriented Bounding Box") and (x is None or y is None or width is None or height is None)) or \
                (self.setup_type == "Segmentation" and segmentation is None):
                    logger.warning(f"Skipping invalid annotation for {anno.get('category_name')} in {image_path}: {anno}")
                    continue

                cursor.execute('''
                    INSERT INTO Annotations (image_id, user_id, type, class_name, x, y, width, height, rotation, segmentation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    user_id,
                    anno_type,
                    anno.get('category_name') or anno.get('label'),
                    x,
                    y,
                    width,
                    height,
                    rotation,
                    segmentation
                ))
                logger.info(f"Saved annotation for {image_path}: type={anno_type}, class={anno.get('category_name')}, bbox=[{x}, {y}, {width}, {height}], rotation={rotation}, user_id={user_id}")

            # Clear preannotations after transfer
            cursor.execute('DELETE FROM Preannotations WHERE image_id = ?', (image_id,))
            logger.info(f"Cleared preannotations for {image_path} after transfer to annotations")
            
            # Update last_modified timestamp for the image
            cursor.execute("UPDATE Images SET last_modified = datetime('now', 'localtime') WHERE image_id = ?", (image_id,))
            
            conn.commit()

    def get_annotations(self, image_path):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (image_path,))
            image_id = cursor.fetchone()
            if not image_id:
                logger.warning(f"No image_id found for path: {image_path} in project {self.name}")
                return []
            image_id = image_id[0]
            cursor.execute('''
                SELECT annotation_id, image_id, type, class_name, x, y, width, height, rotation, segmentation
                FROM Annotations WHERE image_id = ?
            ''', (image_id,))
            annotations = []
            for row in cursor.fetchall():
                anno = {
                    'annotation_id': row[0],
                    'image_id': row[1],
                    'type': 'obbox' if self.setup_type == "Oriented Bounding Box" else row[2],
                    'label': row[3]
                }
                if row[4] is not None and row[5] is not None and row[6] is not None and row[7] is not None:
                    anno['x'] = row[4]
                    anno['y'] = row[5]
                    anno['width'] = row[6]
                    anno['height'] = row[7]
                    anno['bbox'] = [row[4], row[5], row[6], row[7]]
                else:
                    logger.warning(f"Annotation {row[0]} for {image_path} missing bbox coordinates: x={row[4]}, y={row[5]}, width={row[6]}, height={row[7]}")
                if row[8] is not None:
                    anno['rotation'] = row[8]
                else:
                    anno['rotation'] = 0
                if row[9]:
                    try:
                        segmentation = json.loads(row[9])
                        if isinstance(segmentation, list):
                            anno['segmentation'] = [segmentation]
                            anno['points'] = [{'x': segmentation[i], 'y': segmentation[i+1]} for i in range(0, len(segmentation), 2)]
                            anno['closed'] = True
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error parsing segmentation for annotation_id {row[0]}: {e}")
                        anno['segmentation'] = []
                        anno['points'] = []
                annotations.append(anno)
            logger.info(f"Retrieved {len(annotations)} annotations for {image_path} in project {self.name}")
            return annotations
    
    def parse_and_add_annotations(self, temp_upload_dir, image_paths):
        name_matcher = NameMatcher(self.get_classes())
        image_basenames = [os.path.basename(path) for path in image_paths]
        annotation_files = [f for f in os.listdir(temp_upload_dir) if f.endswith(('.json', '.yaml', '.txt'))]

        logger.info(f"Found {len(annotation_files)} annotation files in {temp_upload_dir}: {annotation_files}")
        logger.info(f"Processing {len(image_paths)} images: {image_basenames}")

        for anno_file in [f for f in annotation_files if f.endswith('.json')]:
            anno_path = os.path.join(temp_upload_dir, anno_file)
            try:
                parser = CocoAnnotationParser(anno_path)
                for image_file in image_basenames:
                    annotations = parser.get_annotations_for_image(image_file)
                    if annotations:
                        absolute_image_path = next(
                            (path for path in image_paths if os.path.basename(path).lower() == image_file.lower()), None
                        )
                        if absolute_image_path:
                            normalized_annotations = []
                            for anno in annotations:
                                matched_class = name_matcher.match(anno['category_name'])
                                if matched_class:
                                    normalized_anno = {'category_name': matched_class, 'rotation': 0}
                                    if self.setup_type in ("Bounding Box", "Oriented Bounding Box"):
                                        if anno.get('bbox') and len(anno['bbox']) == 4:
                                            normalized_anno['bbox'] = anno['bbox']
                                        else:
                                            continue
                                    elif self.setup_type == "Segmentation":
                                        if anno.get('segmentation'):
                                            normalized_anno['segmentation'] = anno['segmentation']
                                        else:
                                            continue
                                    normalized_annotations.append(normalized_anno)
                            if normalized_annotations:
                                self.save_annotations(absolute_image_path, normalized_annotations)
            except Exception as e:
                logger.error(f"Error parsing COCO file {anno_file}: {e}")

        yaml_files = [f for f in annotation_files if f.endswith('.yaml')]
        txt_files = set([f for f in annotation_files if f.endswith('.txt')])
        
        if yaml_files:
            yaml_path = os.path.join(temp_upload_dir, yaml_files[0])
            try:
                parser = YoloAnnotationParser(yaml_path, temp_upload_dir)
                for image_file in image_basenames:
                    annotations = parser.get_annotations_for_image(image_file)
                    if annotations:
                        absolute_image_path = next(
                            (path for path in image_paths if os.path.basename(path) == image_file), None
                        )
                        if absolute_image_path:
                            with sqlite3.connect(self.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute('SELECT width, height FROM Images WHERE absolute_path = ?', (absolute_image_path,))
                                result = cursor.fetchone()
                                if result:
                                    img_width, img_height = result
                                    normalized_annotations = []
                                    for anno in annotations:
                                        matched_class = name_matcher.match(anno['category_name'])
                                        if matched_class:
                                            normalized_anno = {'category_name': matched_class, 'rotation': 0}
                                            if self.setup_type == "Bounding Box":
                                                if anno.get('bbox_norm') and len(anno['bbox_norm']) == 4:
                                                    x_center, y_center, w, h = anno['bbox_norm']
                                                    normalized_anno['bbox'] = [
                                                        (x_center - w / 2) * img_width,
                                                        (y_center - h / 2) * img_height,
                                                        w * img_width,
                                                        h * img_height
                                                    ]
                                            elif self.setup_type == "Oriented Bounding Box":
                                                if anno.get('obbox') and len(anno['obbox']) == 8:
                                                    x1, y1, x2, y2, x3, y3, x4, y4 = anno['obbox']
                                                    min_x = min(x1, x2, x3, x4) * img_width
                                                    max_x = max(x1, x2, x3, x4) * img_width
                                                    min_y = min(y1, y2, y3, y4) * img_height
                                                    max_y = max(y1, y2, y3, y4) * img_height
                                                    width = max_x - min_x
                                                    height = max_y - min_y
                                                    dx = (x2 - x1) * img_width
                                                    dy = (y2 - y1) * img_height
                                                    rotation = math.degrees(math.atan2(dy, dx))
                                                    normalized_anno['bbox'] = [min_x, min_y, width, height]
                                                    normalized_anno['rotation'] = rotation
                                            elif self.setup_type == "Segmentation":
                                                if anno.get('segmentation'):
                                                    points = anno['segmentation']
                                                    denormalized_points = []
                                                    for i in range(0, len(points), 2):
                                                        x = points[i] * img_width
                                                        y = points[i + 1] * img_height
                                                        denormalized_points.extend([x, y])
                                                    normalized_anno['segmentation'] = denormalized_points
                                            if 'bbox' in normalized_anno or 'segmentation' in normalized_anno:
                                                normalized_annotations.append(normalized_anno)
                                    if normalized_annotations:
                                        self.save_annotations(absolute_image_path, normalized_annotations, None)
                                else:
                                    logger.error(f"No image dimensions found for {absolute_image_path}")
            except Exception as e:
                logger.error(f"Error initializing YOLO parser with {yaml_files[0]}: {e}")
        else:
            for txt_file in txt_files:
                try:
                    image_file = os.path.splitext(txt_file)[0] + '.jpg'
                    absolute_image_path = next(
                        (path for path in image_paths if os.path.basename(path).lower() == image_file.lower()), None
                    )
                    if absolute_image_path:
                        with open(os.path.join(temp_upload_dir, txt_file), 'r') as f:
                            lines = f.readlines()
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('SELECT width, height FROM Images WHERE absolute_path = ?', (absolute_image_path,))
                            result = cursor.fetchone()
                            if result:
                                img_width, img_height = result
                                project_classes = self.get_classes()
                                normalized_annotations = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        if 0 <= class_id < len(project_classes):
                                            class_name = project_classes[class_id]
                                            if len(parts) == 5:
                                                x_center, y_center, w, h = map(float, parts[1:5])
                                                bbox = [
                                                    (x_center - w / 2) * img_width,
                                                    (y_center - h / 2) * img_height,
                                                    w * img_width,
                                                    h * img_height
                                                ]
                                                normalized_annotations.append({
                                                    'category_name': class_name,
                                                    'bbox': bbox,
                                                    'rotation': 0
                                                })
                                            elif len(parts) == 9 and self.setup_type == "Oriented Bounding Box":
                                                points = list(map(float, parts[1:9]))
                                                min_x = min(points[0::2]) * img_width
                                                max_x = max(points[0::2]) * img_width
                                                min_y = min(points[1::2]) * img_height
                                                max_y = max(points[1::2]) * img_height
                                                width = max_x - min_x
                                                height = max_y - min_y
                                                dx = (points[2] - points[0]) * img_width
                                                dy = (points[3] - points[1]) * img_height
                                                rotation = math.degrees(math.atan2(dy, dx))
                                                normalized_annotations.append({
                                                    'category_name': class_name,
                                                    'bbox': [min_x, min_y, width, height],
                                                    'rotation': rotation
                                                })
                                            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0 and self.setup_type == "Segmentation":
                                                points = list(map(float, parts[1:]))
                                                denormalized_points = []
                                                for i in range(0, len(points), 2):
                                                    denormalized_points.extend([points[i] * img_width, points[i + 1] * img_height])
                                                normalized_annotations.append({
                                                    'category_name': class_name,
                                                    'segmentation': denormalized_points
                                                })
                                if normalized_annotations:
                                    self.save_annotations(absolute_image_path, normalized_annotations)
                except Exception as e:
                    logger.error(f"Error parsing standalone TXT file {txt_file}: {e}")
                    
    def get_image_count(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM Images')
            count = cursor.fetchone()[0]
            logger.info(f"Image count for project {self.name}: {count}")
            return count

    def get_annotated_image_count(self):
        """获取有标注记录或已审核的图片数量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 统计已标注（在Annotations中）或已审核（在ReviewedImages中）的图片
            # 同时必须在 Images 表中存在（避免统计已删除图片的残留记录）
            cursor.execute('''
                SELECT COUNT(DISTINCT image_id) 
                FROM (
                    SELECT image_id FROM Annotations
                    UNION
                    SELECT image_id FROM ReviewedImages
                )
                WHERE image_id IN (SELECT image_id FROM Images)
            ''')
            count = cursor.fetchone()[0]
            logger.info(f"Annotated/Reviewed image count for project {self.name}: {count}")
            return count

    def get_completed_image_count(self):
        """获取已完成（已保存/审核）的有效图片数量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 必须关联 Images 表，避免统计已删除图片的残留记录
            cursor.execute('''
                SELECT COUNT(DISTINCT r.image_id) 
                FROM ReviewedImages r
                INNER JOIN Images i ON r.image_id = i.image_id
            ''')
            count = cursor.fetchone()[0]
            logger.info(f"Completed/reviewed image count for project {self.name}: {count}")
            return count

    def get_class_distribution(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT class_name, COUNT(*) FROM Annotations GROUP BY class_name')
            distribution = dict(cursor.fetchall())
            logger.info(f"Class distribution for project {self.name}: {distribution}")
            return distribution

    def get_annotations_per_image(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(a.annotation_id)
                FROM Images i
                LEFT JOIN Annotations a ON i.image_id = a.image_id
                GROUP BY i.image_id
            ''')
            counts = [row[0] for row in cursor.fetchall()]
            #logger.info(f"Annotations per image for project {self.name}: {counts}")
            return counts

    def get_annotated_images(self):
        """获取所有已标注的图像信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT i.image_id, i.absolute_path, i.width, i.height,
                       LENGTH(TRIM(COALESCE(i.absolute_path, ''))) > 0 as has_path
                FROM Images i
                INNER JOIN Annotations a ON i.image_id = a.image_id
                ORDER BY i.image_id
            ''')
            annotated_images = []
            for row in cursor.fetchall():
                if row[4]:  # has_path check
                    annotated_images.append({
                        'id': row[0],
                        'path': row[1],
                        'name': os.path.basename(row[1]) if row[1] else f'image_{row[0]}',
                        'width': row[2],
                        'height': row[3]
                    })
            logger.info(f"Retrieved {len(annotated_images)} annotated images for project {self.name}")
            logger.info(f"Retrieved {len(annotated_images)} annotated images for project {self.name}")
            return annotated_images

    def get_completed_images(self):
        """获取所有已完成（已审核）的图像信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # ReviewedImages table tracks all saved/completed images (empty or not)
            # Use LEFT JOIN to ensure we get path info from Images
            cursor.execute('''
                SELECT DISTINCT i.image_id, i.absolute_path, i.width, i.height,
                       LENGTH(TRIM(COALESCE(i.absolute_path, ''))) > 0 as has_path
                FROM ReviewedImages r
                JOIN Images i ON r.image_id = i.image_id
            ''')
            completed_images = []
            for row in cursor.fetchall():
                if row[4]:  # has_path
                    completed_images.append({
                        'id': row[0],
                        'path': row[1],
                        'name': os.path.basename(row[1]) if row[1] else f'image_{row[0]}',
                        'width': row[2],
                        'height': row[3]
                    })
            logger.info(f"Retrieved {len(completed_images)} reviewed/completed images for project {self.name}")
            return completed_images

    def get_annotations_by_image_id(self, image_id):
        """根据图像ID获取标注信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT annotation_id, type, class_name, x, y, width, height, rotation, segmentation
                FROM Annotations WHERE image_id = ?
            ''', (image_id,))
            annotations = []
            for row in cursor.fetchall():
                anno = {
                    'annotation_id': row[0],
                    'type': row[1],
                    'class_name': row[2],
                    'x': row[3],
                    'y': row[4],
                    'width': row[5],
                    'height': row[6],
                    'rotation': row[7] or 0
                }
                if row[8]:  # segmentation
                    try:
                        segmentation = json.loads(row[8])
                        if isinstance(segmentation, list):
                            anno['points'] = [{'x': segmentation[i], 'y': segmentation[i+1]} 
                                            for i in range(0, len(segmentation), 2)]
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error parsing segmentation for annotation {row[0]}: {e}")
                        anno['points'] = []
                else:
                    anno['points'] = []
                annotations.append(anno)
            return annotations

    def get_video_annotations(self, video_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.frame_number, f.timestamp, f.image_id
                FROM Frames f
                WHERE f.video_id = ?
                ORDER BY f.frame_number
            ''', (video_id,))
            frames = cursor.fetchall()
            if not frames:
                return {}

            image_ids = [row[2] for row in frames]
            placeholders = ','.join('?' for _ in image_ids)

            cursor.execute(f'''
                SELECT image_id, annotation_id, type, class_name, x, y, width, height, rotation, segmentation
                FROM Annotations
                WHERE image_id IN ({placeholders})
            ''', image_ids)
            ann_rows = cursor.fetchall()

            cursor.execute(f'''
                SELECT image_id, preannotation_id, type, class_name, x, y, width, height, rotation, segmentation, confidence
                FROM Preannotations
                WHERE image_id IN ({placeholders})
            ''', image_ids)
            pre_rows = cursor.fetchall()

            ann_map = {}
            for row in ann_rows:
                image_id = row[0]
                ann_map.setdefault(image_id, []).append({
                    'id': row[1],
                    'type': row[2],
                    'label': row[3],
                    'x': row[4],
                    'y': row[5],
                    'width': row[6],
                    'height': row[7],
                    'rotation': row[8] or 0,
                    'segmentation': row[9],
                    'isPreannotation': False
                })

            pre_map = {}
            for row in pre_rows:
                image_id = row[0]
                pre_map.setdefault(image_id, []).append({
                    'id': row[1],
                    'type': row[2],
                    'label': row[3],
                    'x': row[4],
                    'y': row[5],
                    'width': row[6],
                    'height': row[7],
                    'rotation': row[8] or 0,
                    'segmentation': row[9],
                    'confidence': row[10] if row[10] is not None else 1.0,
                    'isPreannotation': True
                })

            result = {}
            for frame_number, timestamp, image_id in frames:
                frame_annotations = []
                for item in ann_map.get(image_id, []):
                    frame_annotations.append(item)
                for item in pre_map.get(image_id, []):
                    frame_annotations.append(item)
                result[str(frame_number)] = {
                    'timestamp': timestamp,
                    'annotations': frame_annotations
                }
            return result

    def commit_preannotations_for_video(self, video_id, user_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT f.image_id
                    FROM Frames f
                    WHERE f.video_id = ?
                ''', (video_id,))
                image_ids = [row[0] for row in cursor.fetchall()]

                if not image_ids:
                    return {'success': False, 'error': f'No frames found for video {video_id}'}

                placeholders = ','.join('?' for _ in image_ids)
                cursor.execute(f'''
                    INSERT INTO Annotations (image_id, user_id, type, class_name, x, y, width, height, rotation, segmentation)
                    SELECT image_id, ?, type, class_name, x, y, width, height, rotation, segmentation
                    FROM Preannotations
                    WHERE image_id IN ({placeholders})
                ''', [user_id] + image_ids)
                transferred = cursor.rowcount

                if transferred > 0:
                    cursor.execute(f'DELETE FROM Preannotations WHERE image_id IN ({placeholders})', image_ids)
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO ReviewedImages (image_id, user_id)
                        SELECT DISTINCT image_id, ? FROM Annotations WHERE image_id IN ({placeholders})
                    ''', [user_id] + image_ids)

                conn.commit()
                return {'success': True, 'transferred': transferred}
        except Exception as error:
            return {'success': False, 'error': str(error)}

    def delete_annotations(self, image_path=None, user_id=None, video_id=None, start_frame=None, end_frame=None, unmark_reviewed=True):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            image_ids = []
            if image_path:
                cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (image_path,))
                row = cursor.fetchone()
                if row:
                    image_ids = [row[0]]
            elif video_id and start_frame is not None and end_frame is not None:
                cursor.execute('''
                    SELECT f.image_id
                    FROM Frames f
                    WHERE f.video_id = ? AND f.frame_number >= ? AND f.frame_number <= ?
                ''', (video_id, start_frame, end_frame))
                image_ids = [row[0] for row in cursor.fetchall()]
            elif video_id:
                cursor.execute('SELECT image_id FROM Frames WHERE video_id = ?', (video_id,))
                image_ids = [row[0] for row in cursor.fetchall()]

            if image_ids:
                placeholders = ','.join('?' for _ in image_ids)
                if user_id:
                    cursor.execute(f'DELETE FROM Annotations WHERE image_id IN ({placeholders}) AND user_id = ?', image_ids + [user_id])
                else:
                    cursor.execute(f'DELETE FROM Annotations WHERE image_id IN ({placeholders})', image_ids)
                annotations_deleted = cursor.rowcount

                reviewed_deleted = 0
                if unmark_reviewed:
                    cursor.execute(f'DELETE FROM ReviewedImages WHERE image_id IN ({placeholders})', image_ids)
                    reviewed_deleted = cursor.rowcount
            else:
                if user_id:
                    cursor.execute('DELETE FROM Annotations WHERE user_id = ?', (user_id,))
                else:
                    cursor.execute('DELETE FROM Annotations')
                annotations_deleted = cursor.rowcount
                reviewed_deleted = 0
                if unmark_reviewed:
                    cursor.execute('DELETE FROM ReviewedImages')
                    reviewed_deleted = cursor.rowcount

            conn.commit()
            return {'annotations_deleted': annotations_deleted, 'reviewed_deleted': reviewed_deleted}

    def delete_preannotations(self, image_path=None, video_id=None, start_frame=None, end_frame=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            image_ids = []
            if image_path:
                cursor.execute('SELECT image_id FROM Images WHERE absolute_path = ?', (image_path,))
                row = cursor.fetchone()
                if row:
                    image_ids = [row[0]]
            elif video_id and start_frame is not None and end_frame is not None:
                cursor.execute('''
                    SELECT f.image_id
                    FROM Frames f
                    WHERE f.video_id = ? AND f.frame_number >= ? AND f.frame_number <= ?
                ''', (video_id, start_frame, end_frame))
                image_ids = [row[0] for row in cursor.fetchall()]
            elif video_id:
                cursor.execute('SELECT image_id FROM Frames WHERE video_id = ?', (video_id,))
                image_ids = [row[0] for row in cursor.fetchall()]

            if image_ids:
                placeholders = ','.join('?' for _ in image_ids)
                cursor.execute(f'DELETE FROM Preannotations WHERE image_id IN ({placeholders})', image_ids)
                deleted = cursor.rowcount
            else:
                cursor.execute('DELETE FROM Preannotations')
                deleted = cursor.rowcount

            conn.commit()
            return deleted

    # ==================== CT 体数据管理 ====================

    def add_ct_volume(self, volume_path: str, preview_path: str = None,
                      nii_path: str = None, csv_path: str = None) -> 'int | None':
        """
        解析 MHD 或 NIfTI 文件并将体数据元信息写入 CTVolumes 表。
        可选传入配套的 NIfTI 分割掩码路径和 CSV 标注路径。
        返回 volume_id。
        """
        try:
            from medseg.utils.ct_utils import parse_volume_metadata
            meta = parse_volume_metadata(volume_path)
        except Exception as e:
            logger.error(f"CT 元数据解析失败 {volume_path}: {e}")
            return None

        name = os.path.basename(volume_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO CTVolumes
                    (absolute_path, name, shape_x, shape_y, shape_z,
                     spacing_x, spacing_y, spacing_z, preview_path, nii_path, csv_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                volume_path, name,
                meta['shape_x'], meta['shape_y'], meta['shape_z'],
                meta['spacing_x'], meta['spacing_y'], meta['spacing_z'],
                preview_path, nii_path, csv_path
            ))
            conn.commit()
            cursor.execute('SELECT volume_id FROM CTVolumes WHERE absolute_path = ?', (volume_path,))
            row = cursor.fetchone()
            if row:
                logger.info(f"CT 体数据已添加: {volume_path}, volume_id={row[0]}")
                return row[0]
        return None

    def add_ct_annotations_from_csv(self, volume_id: int, csv_path: str,
                                    series_id: str = None) -> int:
        """
        从 CSV 文件读取结节标注并写入 CTAnnotations 表。
        支持的 CSV 格式（自动检测列名，大小写不敏感）：

          LUNA16 格式（标准）：
            seriesuid, coordX, coordY, coordZ, diameter_mm

          LIDC/自定义格式（备选列名）：
            - X坐标列：coordX / x / coord_x / world_x / cx
            - Y坐标列：coordY / y / coord_y / world_y / cy
            - Z坐标列：coordZ / z / coord_z / world_z / cz
            - 直径列：diameter_mm / diameter / diam / radius_mm（自动×2）
            - 系列ID列：seriesuid / series_uid / uid / id / case / filename
            - 标签列：label / class / type / nodule_type

        参数：
            volume_id  : CTVolumes 表中的 volume_id
            csv_path   : CSV 文件路径
            series_id  : 本体数据的系列ID（MHD文件名去扩展名），用于过滤多病例CSV
        返回：写入的标注条数
        """
        import csv as csv_mod

        if not os.path.isfile(csv_path):
            logger.warning(f"CSV 文件不存在: {csv_path}")
            return 0

        # 候选列名映射（小写）
        _X_COLS  = ['coordx', 'x', 'coord_x', 'world_x', 'cx', 'pos_x']
        _Y_COLS  = ['coordy', 'y', 'coord_y', 'world_y', 'cy', 'pos_y']
        _Z_COLS  = ['coordz', 'z', 'coord_z', 'world_z', 'cz', 'pos_z']
        _D_COLS  = ['diameter_mm', 'diameter', 'diam', 'size_mm', 'size']
        _R_COLS  = ['radius_mm', 'radius']  # 半径→直径×2
        _ID_COLS = ['seriesuid', 'series_uid', 'uid', 'id', 'case', 'filename', 'patient_id']
        _LB_COLS = ['label', 'class', 'type', 'nodule_type', 'category']

        def _find_col(header_lower, candidates):
            for c in candidates:
                if c in header_lower:
                    return header_lower.index(c)
            return None

        records = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv_mod.reader(f)
                raw_header = next(reader)
                header_lower = [h.strip().lower() for h in raw_header]

                ix = _find_col(header_lower, _X_COLS)
                iy = _find_col(header_lower, _Y_COLS)
                iz = _find_col(header_lower, _Z_COLS)
                id_col = _find_col(header_lower, _ID_COLS)
                d_col  = _find_col(header_lower, _D_COLS)
                r_col  = _find_col(header_lower, _R_COLS)
                lb_col = _find_col(header_lower, _LB_COLS)

                if ix is None or iy is None or iz is None:
                    logger.error(f"CSV 中找不到坐标列: {raw_header}")
                    return 0

                for row in reader:
                    if not row or all(v.strip() == '' for v in row):
                        continue
                    # 系列ID过滤
                    if series_id and id_col is not None:
                        row_uid = str(row[id_col]).strip()
                        # 支持 seriesuid 完整匹配或 文件名匹配
                        if row_uid != series_id and not series_id.startswith(row_uid) and not row_uid.startswith(series_id):
                            continue
                    try:
                        cx = float(row[ix])
                        cy = float(row[iy])
                        cz = float(row[iz])
                    except (ValueError, IndexError):
                        continue

                    diam = 0.0
                    if d_col is not None and d_col < len(row):
                        try: diam = float(row[d_col])
                        except ValueError: pass
                    elif r_col is not None and r_col < len(row):
                        try: diam = float(row[r_col]) * 2
                        except ValueError: pass

                    label = 'nodule'
                    if lb_col is not None and lb_col < len(row):
                        label = str(row[lb_col]).strip() or 'nodule'

                    records.append((volume_id, label, cx, cy, cz, diam, 'csv'))
        except Exception as e:
            logger.error(f"读取 CSV 失败 {csv_path}: {e}")
            return 0

        if not records:
            logger.warning(f"CSV 无匹配记录 series_id={series_id}: {csv_path}")
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 清除该体数据旧的 CSV 标注
            cursor.execute("DELETE FROM CTAnnotations WHERE volume_id=? AND source='csv'", (volume_id,))
            cursor.executemany('''
                INSERT INTO CTAnnotations (volume_id, label, coord_x, coord_y, coord_z, diameter_mm, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            logger.info(f"CT 标注已写入 volume_id={volume_id}: {len(records)} 条")
            return len(records)

    def get_ct_annotations_for_volume(self, volume_id: int) -> list:
        """返回某体数据的全部标注（世界坐标，包含临床特征字段）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(CTAnnotations)")
            cols = [c[1] for c in cursor.fetchall()]
            if all(x in cols for x in ('location', 'texture', 'risk_level', 'signs')):
                cursor.execute('''
                    SELECT anno_id, label, coord_x, coord_y, coord_z, diameter_mm, location, texture, risk_level, signs
                    FROM CTAnnotations WHERE volume_id = ?
                ''', (volume_id,))
                return [{
                    'anno_id': r[0], 'label': r[1],
                    'coord_x': r[2], 'coord_y': r[3], 'coord_z': r[4],
                    'diameter_mm': r[5],
                    'location': r[6],
                    'texture': r[7],
                    'risk_level': r[8],
                    'signs': r[9]
                } for r in cursor.fetchall()]
            else:
                cursor.execute('''
                    SELECT anno_id, label, coord_x, coord_y, coord_z, diameter_mm
                    FROM CTAnnotations WHERE volume_id = ?
                ''', (volume_id,))
                return [{
                    'anno_id': r[0], 'label': r[1],
                    'coord_x': r[2], 'coord_y': r[3], 'coord_z': r[4],
                    'diameter_mm': r[5],
                    'location': None,
                    'texture': None,
                    'risk_level': None,
                    'signs': None
                } for r in cursor.fetchall()]

    def get_ct_volumes(self) -> list:
        """返回项目内所有 CT 体数据列表（含 nii_path / csv_path）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT volume_id, absolute_path, name,
                       shape_x, shape_y, shape_z,
                       spacing_x, spacing_y, spacing_z,
                       preview_path, added_at, nii_path, csv_path
                FROM CTVolumes ORDER BY added_at DESC
            ''')
            rows = cursor.fetchall()
            return [{
                'volume_id': r[0],
                'absolute_path': r[1],
                'name': r[2],
                'shape_x': r[3], 'shape_y': r[4], 'shape_z': r[5],
                'spacing_x': r[6], 'spacing_y': r[7], 'spacing_z': r[8],
                'preview_path': r[9],
                'added_at': r[10],
                'nii_path': r[11],
                'csv_path': r[12],
            } for r in rows]

    def get_ct_volume_path(self, volume_id: int) -> str | None:
        """根据 volume_id 返回 MHD 文件绝对路径"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT absolute_path FROM CTVolumes WHERE volume_id = ?', (volume_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_ct_volume(self, volume_id: int) -> dict | None:
        """根据 volume_id 返回单个 CT 体数据完整信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT volume_id, absolute_path, name,
                       shape_x, shape_y, shape_z,
                       spacing_x, spacing_y, spacing_z,
                       preview_path, added_at, nii_path, csv_path
                FROM CTVolumes
                WHERE volume_id = ?
            ''', (volume_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                'volume_id': row[0],
                'absolute_path': row[1],
                'name': row[2],
                'shape_x': row[3], 'shape_y': row[4], 'shape_z': row[5],
                'spacing_x': row[6], 'spacing_y': row[7], 'spacing_z': row[8],
                'preview_path': row[9],
                'added_at': row[10],
                'nii_path': row[11],
                'csv_path': row[12],
            }

    def update_ct_volume_nii(self, volume_id: int, nii_path: str):
        """更新体数据的 NIfTI 掩码路径"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE CTVolumes SET nii_path = ? WHERE volume_id = ?', (nii_path, volume_id))
            conn.commit()

    def mark_ct_volume_reviewed(self, volume_id: int, user_id: int | None = None):
        """标记 CT 体数据已完成一次显式保存/确认。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO CTReviewedVolumes (volume_id, reviewed_at, user_id)
                VALUES (?, CURRENT_TIMESTAMP, ?)
            ''', (volume_id, user_id))
            conn.commit()

    def get_ct_reviewed_volume_ids(self) -> set[int]:
        """返回已确认保存的 CT volume_id 集合。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT volume_id FROM CTReviewedVolumes')
            return {int(row[0]) for row in cursor.fetchall()}

    def get_ct_reviewed_count(self) -> int:
        """返回已确认保存的 CT 数量。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM CTReviewedVolumes')
            row = cursor.fetchone()
            return int(row[0] or 0)

    def delete_ct_volumes(self, volume_ids: list[int]) -> int:
        """删除指定 CT 体数据及其关联标注，并尽量清理项目内文件。"""
        safe_ids = []
        for volume_id in volume_ids or []:
            try:
                safe_ids.append(int(volume_id))
            except (TypeError, ValueError):
                continue

        if not safe_ids:
            return 0

        deleted = 0
        project_root = os.path.abspath(self.project_path)

        def _is_project_file(path: str) -> bool:
            if not path:
                return False
            try:
                return os.path.commonpath([project_root, os.path.abspath(path)]) == project_root
            except Exception:
                return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in safe_ids)
            cursor.execute(f'''
                SELECT volume_id, absolute_path, preview_path, nii_path
                FROM CTVolumes
                WHERE volume_id IN ({placeholders})
            ''', tuple(safe_ids))
            rows = cursor.fetchall()

            for volume_id, absolute_path, preview_path, nii_path in rows:
                cursor.execute('DELETE FROM CTAnnotations WHERE volume_id = ?', (volume_id,))
                cursor.execute('DELETE FROM CTVolumes WHERE volume_id = ?', (volume_id,))
                deleted += 1

                for file_path in (preview_path, nii_path, absolute_path):
                    if file_path and _is_project_file(file_path) and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except IsADirectoryError:
                            shutil.rmtree(file_path, ignore_errors=True)
                        except Exception as error:
                            logger.warning(f"删除 CT 文件失败 {file_path}: {error}")

                abs_lower = str(absolute_path or '').lower()
                if abs_lower.endswith('.mhd') and _is_project_file(absolute_path):
                    try:
                        with open(absolute_path, 'r', encoding='utf-8', errors='ignore') as fh:
                            for line in fh:
                                if line.strip().startswith('ElementDataFile'):
                                    raw_name = line.split('=', 1)[1].strip()
                                    raw_path = os.path.join(os.path.dirname(absolute_path), raw_name)
                                    if _is_project_file(raw_path) and os.path.exists(raw_path):
                                        os.remove(raw_path)
                                    break
                    except Exception as error:
                        logger.warning(f"删除 CT 配套 RAW 文件失败 {absolute_path}: {error}")

            conn.commit()

        return deleted

    def get_ct_volume_count(self) -> int:
        """返回 CT 体数据个数"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM CTVolumes')
            return cursor.fetchone()[0]
