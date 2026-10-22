"""
app.py - live dashboard for the next-session direction model.

Routes
  /          live prediction for the next session, out-of-sample charts, backtest
  /track     every prediction this deployment has made, scored against reality
  /api/...   the same payloads as JSON, so the page can refresh without a reload

The old build asked the user to type Open/High/Low/Close/Volume into a form. The
model now needs 25 features derived from ~50 sessions of history, so the app
fetches real data instead. /predict/ is kept only to redirect old links.
"""

from flask import Flask, render_template, redirect, url_for, jsonify, request

import utils

app = Flask(__name__)
utils.init_db()


def _dashboard_payload(force=False):
    """Everything the dashboard renders. Errors are surfaced, not swallowed."""
    payload = {
        'info': utils.model_info(),
        'chart': utils.chart_history(90),
        'backtest': utils.backtest_summary(),
        'prediction': None,
        'error': None,
    }
    try:
        prediction = utils.predict_latest(force=force)
        # Log before rendering, so what the user sees is what gets scored later.
        utils.log_prediction(prediction)
        payload['prediction'] = prediction
    except Exception as exc:
        payload['error'] = 'Could not build a live prediction: ' + str(exc)

    payload['track'] = utils.track_record(limit=10)
    return payload


@app.route('/')
def main():
    force = request.args.get('refresh') == '1'
    return render_template('index.html', **_dashboard_payload(force=force))


@app.route('/track')
def track():
    return render_template('track.html',
                           track=utils.track_record(limit=200),
                           info=utils.model_info())


@app.route('/api/prediction')
def api_prediction():
    try:
        p = utils.predict_latest(force=request.args.get('refresh') == '1')
        utils.log_prediction(p)
        return jsonify(p)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 503


@app.route('/api/track')
def api_track():
    return jsonify(utils.track_record(limit=200))


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # Manual OHLCV entry cannot produce a prediction any more - the features need
    # history. Old bookmarks land on the live dashboard instead.
    return redirect(url_for('main'))


@app.errorhandler(404)
def error(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
