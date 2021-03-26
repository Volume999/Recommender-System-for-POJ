from flask import Flask, jsonify, request
from Main import engine, user_data
import json

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'


@app.route('/user/testing', methods=['GET'])
def execute():
    return jsonify(dict(engine.execute_for_user(user_data)))


@app.route('/user/<string:name>')
def execute_user(name):
    return jsonify(dict(engine.data.users[name].recommendations))


@app.route('/newuser', methods=['POST'])
def newuser():
    data = json.loads(request.get_json())
    print(data)
    print([(p, v, t) for [p, v, t] in data])
    return jsonify(dict(engine.execute_for_user(data)))


if __name__ == '__main__':
    app.run(debug=True)
