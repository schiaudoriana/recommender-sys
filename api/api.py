import flask
from flask import jsonify

from service.service_class import Service

app = flask.Flask(__name__)
app.config["DEBUG"] = True

service = Service()


@app.route('/recommend/<string:description>/<string:tags>', methods=['GET'])
def give_recommendations(description, tags):
    result = service.give_recommendations(description, tags, 0)
    return jsonify(result)


app.run()
