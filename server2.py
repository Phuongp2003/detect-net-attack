from flask import Flask

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def data():
    return {"message": "This is a test endpoint"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
