from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/get_json", methods=['GET'])
def get_json():
    data = {"name" : "TEST", "context" : 1234}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=false)



