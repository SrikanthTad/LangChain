from flask import Flask, render_template, request, jsonify
from main import ice_break

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    person_info, profile_pic_url = ice_break(
        name=name
    )
    return jsonify(
        {
            "summary": person_info.summary,
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
