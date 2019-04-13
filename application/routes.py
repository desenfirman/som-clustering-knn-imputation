from flask import Blueprint, render_template, request, redirect, url_for
from pandas import read_csv


mod = Blueprint('main', __name__,
                template_folder='templates',
                static_folder='static')


@mod.route('/')
def home():
    return render_template("main.html")


@mod.route('/clustering', methods=['GET', 'POST'])
def mulai_clustering():
    data = dict()
    if request.method == 'POST':
        data['k_value'] = request.form['k_value']
        data['neuron_width'] = request.form['neuron_width']
        data['neuron_height'] = request.form['neuron_height']
        data['alpha'] = request.form['alpha']
        data['eta'] = request.form['eta']
        data['epoch'] = request.form['epoch']
        if request.form['pengukuran'] == 'qe':
            data['tipe_pengukuran'] = 'Quantization Error'
        elif request.form['pengukuran'] == 'dbi':
            data['tipe_pengukuran'] = 'Davies-Bouldin Index'
        return render_template("clustering.html", data=data)
    else:
        return redirect(url_for('main.home'))
