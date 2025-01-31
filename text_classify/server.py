import os

from flask import Flask, jsonify, request

from predictor import Predictor

__file_dir__ = os.path.abspath(os.path.dirname(__file__))
print(f"当前server.py所在的文件夹:{__file_dir__}")

app = Flask(__name__)
predictor = Predictor(
    mod_path=os.path.join(__file_dir__, "mod.pt"),
    token_path=os.path.join(__file_dir__, "tokens.json")
)


@app.route("/")
def index():
    return "欢迎使用文本分类模型!"


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            # Get请求的参数获取方式
            text = request.args.get('text', None)
        else:
            # POST请求的参数获取方式
            text = request.form.get('text', None)
        _r = predictor.predict(text=text)
        return jsonify(_r)
    except Exception as e:
        return jsonify({'code': 1, 'msg': f'服务器异常:{e}'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)
