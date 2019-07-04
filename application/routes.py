from flask import Blueprint, session, render_template, request, redirect, url_for, Response, stream_with_context
from pprint import pprint
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
    new_columns = dataset.columns.values
    new_columns[0] = 'First_Column'
    dataset.columns = new_columns
    dataset = dataset.set_index('First_Column')
    session['raw_dataset'] = dataset
    pprint(dataset)
    return session['raw_dataset'].iloc[:, :].values


@mod.route('/clustering', methods=['GET', 'POST'])
def mulai_clustering():
    if request.method == 'POST':
        session['k_value'] = int(request.form['k_value'])

        file_handler = request.files['input_dataset']
        dataset_unnormalized = extract_dataset(file_handler)
        dataset_unnormalized = knn_imputation.impute_dataset(
            dataset_unnormalized, session['k_value'])
        session['dataset'] = self_organizing_maps.normalize_data(
            dataset_unnormalized)

        session['input_dataset_filename'] = file_handler.filename
        session['neuron_width'] = int(request.form['neuron_width'])
        session['neuron_height'] = int(request.form['neuron_height'])
        session['alpha_0'] = float(request.form['alpha'])
        session['eta_0'] = float(request.form['eta'])
        session['max_epoch'] = int(request.form['epoch'])

        attr_size = len(session['dataset'][0])
        session['weight'] = self_organizing_maps.init_som_net(
            session['neuron_height'], session['neuron_width'], attr_size)
        # pprint(session['dataset'])
        return render_template("clustering.html")
    else:
        return redirect(url_for('main.home'))


@mod.route('/training_progress')
def training_progress():
    @stream_with_context
    def training():
        t = 1

        while(t <= session['max_epoch']):
            plot_url = '0'
            if t == session['max_epoch']:
                plot_url = self_organizing_maps.silhouette_visualizer(session['weight'], session['dataset'])
            alpha_t = session['alpha_0'] * (1 / t)
            eta_t = session['eta_0'] * exp(-1 * (t / session['max_epoch']))

            session['weight'] = self_organizing_maps.one_epoch_training(
                session['dataset'], session['weight'], alpha_t, eta_t)

            cluster_visualization = cluster_visualization_in_JSON(
                session['weight'], session['dataset'])

            epoch_t_response = '"epoch": ' + str(t) + ', '
            weight_response = '"weight": ' + str(session['weight']) + ', '
            cluster_vis_t_response = '"cluster": ' + \
                cluster_visualization + ', '
            score_t_response = '"score": ' + \
                str(get_score(session['weight'],
                              session['dataset'])) + ', '
            plot_url_response = '"plot_url": "' + \
                plot_url + '", '
            jenis_pengukuran_response = '"jenis":"' + \
                str('Avg Silhouette Coefficient') + '" '

            yield "data: {" + epoch_t_response + weight_response + \
                cluster_vis_t_response + \
                score_t_response + plot_url_response + jenis_pengukuran_response + "}\n\n"

            t += 1
            time.sleep(0.5)

    return Response(training(), content_type='text/event-stream')


def cluster_visualization_in_JSON(training_weight, input_dataset):
    indexes = list(session['raw_dataset'].index)
    list_cluster = '['
    for i in range(0, len(input_dataset)):
        index = '"' + indexes[i] + '"'
        bmu = self_organizing_maps.penentuan_cluster(
            training_weight, input_dataset[i])
        json = '{"id": ' + str(index) + ', "neuron": [' + \
            str(bmu[0]) + ', ' + str(bmu[1]) + ']}'
        list_cluster += json
        if i != len(input_dataset) - 1:
            list_cluster += ', '
    return list_cluster + ']'


def get_score(weight, dataset_input):
    sil = self_organizing_maps.average_silhouette(weight, dataset_input)
    return sil


@mod.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
