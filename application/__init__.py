# __init__.py
import os
from flask import Flask
from .routes import mod as main_blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_blueprint)
    app.config.from_pyfile(os.path.join(app.root_path, 'config.py'))
    return app
