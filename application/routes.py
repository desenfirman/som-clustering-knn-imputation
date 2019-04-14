from flask import Blueprint, session, render_template, request, redirect, url_for, Response, stream_with_context
import time
from math import exp
from pandas import read_csv
from main_algorithm import self_organizing_maps, knn_imputation

mod = Blueprint('main', __name__,
                template_folder='templates',
                static_folder='static')


@mod.route('/')
def home():
    return render_template("main.html")


def extract_dataset(file):
    dataset = read_csv(file)
    dataset_value = dataset.iloc[:, :].values
    return dataset_value


@mod.route('/clustering', methods=['GET', 'POST'])
def mulai_clustering():
    if request.method == 'POST':
        file_handler = request.files['input_dataset']
        dataset_unnormalized = extract_dataset(file_handler)
        session['dataset'] = self_organizing_maps.normalize_data(
            dataset_unnormalized)

        session['input_dataset_filename'] = file_handler.filename
        session['k_value'] = int(request.form['k_value'])
        session['neuron_width'] = int(request.form['neuron_width'])
        session['neuron_height'] = int(request.form['neuron_height'])
        session['alpha_0'] = float(request.form['alpha'])
        session['eta_0'] = float(request.form['eta'])
        session['max_epoch'] = int(request.form['epoch'])
        if request.form['pengukuran'] == 'qe':
            session['tipe_pengukuran'] = 'Quantization Error'
        elif request.form['pengukuran'] == 'dbi':
            session['tipe_pengukuran'] = 'Davies-Bouldin Index'

        # start session
        session['pengukuran'] = request.form['pengukuran']

        attr_size = len(session['dataset'][0])
        session['weight'] = self_organizing_maps.init_som_net(
            session['neuron_height'], session['neuron_width'], attr_size)

        return render_template("clustering.html")
    else:
        return redirect(url_for('main.home'))


@mod.route('/training_progress')
def training_progress():
    @stream_with_context
    def training():
        t = 1

        while(t <= session['max_epoch']):
            alpha_t = session['alpha_0'] * (1 / t)
            eta_t = session['eta_0'] * exp(-1 * (t / session['max_epoch']))

            epoch_t_response = '"epoch": ' + str(t) + ', '
            weight_response = '"weight": ' + str(session['weight']) + ', '
            score_t_response = '"score": ' + \
                str(get_score(session['weight'],
                              session['dataset'])) + ', '
            jenis_pengukuran_response = '"jenis":"' + \
                str(session['tipe_pengukuran']) + '" '

            yield "data: {" + epoch_t_response + weight_response + \
                score_t_response + jenis_pengukuran_response + "}\n\n"

            session['weight'] = self_organizing_maps.one_epoch_training(
                session['dataset'], session['weight'], alpha_t, eta_t)
            t += 1
            time.sleep(0.5)

    return Response(training(), content_type='text/event-stream')


def get_score(weight, dataset_input):
    if session['pengukuran'] == "qe":
        return self_organizing_maps.quantization_error(weight, dataset_input)
    elif session['pengukuran'] == "dbi":
        return self_organizing_maps.davies_bouldin_index(weight, dataset_input)
