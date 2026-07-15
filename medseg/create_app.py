from flask import Flask, send_from_directory
from flask_login import LoginManager
from medseg.config import Config
from medseg.models.user import init_db, get_user_by_id, User
import os
import mimetypes


def _load_blueprint(module_path: str):
    module = __import__(module_path, fromlist=['bp'])
    return module.bp


def create_app(config_object=None):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)
    if config_object:
        app.config.from_object(config_object)
    app.secret_key = app.config['SECRET_KEY']
    app.config['MAX_CONTENT_LENGTH'] = app.config.get('MAX_CONTENT_LENGTH', 500 * 1024 * 1024)
    app.config['MAX_FORM_PARTS'] = app.config.get('MAX_FORM_PARTS', 1000000)

    # 修复静态文件 MIME 类型
    mimetypes.add_type('application/javascript', '.js')
    mimetypes.add_type('text/css', '.css')
    mimetypes.add_type('application/json', '.json')

    projects_folder = app.config['PROJECTS_FOLDER']
    os.makedirs(projects_folder, exist_ok=True)

    with app.app_context():
        init_db()

    # 只注册 auth、dashboard（CT版）、ct_viewer
    for module_path in [
        'medseg.routes.auth',
        'medseg.routes.dashboard',
        'medseg.routes.ct_viewer',
    ]:
        app.register_blueprint(_load_blueprint(module_path))

    @app.route('/projects/<path:filename>')
    def serve_project_file(filename):
        return send_from_directory(app.config['PROJECTS_FOLDER'], filename)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        user_data = get_user_by_id(user_id)
        if user_data:
            return User(
                user_data[0], user_data[1],
                user_data[3], user_data[4],
                user_data[5], user_data[6],
                user_data[7] if len(user_data) > 7 else 'admin'
            )
        return None

    return app
