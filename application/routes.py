from flask import Blueprint, render_template


mod = Blueprint('main', __name__,
                template_folder='templates',
                static_folder='static')


@mod.route('/')
def home():
    return render_template("main.html")
