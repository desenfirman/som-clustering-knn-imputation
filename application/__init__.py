# __init__.py
import os
from flask import Flask
from flask_session import Session
from .routes import mod as main_blueprint

sess = Session()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main_blueprint)
    app.config.from_pyfile(os.path.join(app.root_path, 'config.py'))
    sess.init_app(app)
    return app
