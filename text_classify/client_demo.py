import requests

text = "质量还不错，非常喜欢"
url = "http://localhost:9999/predict"

response = requests.post(url, data={'text': text})
# response = requests.get(url, params={'text': text})
if response.status_code == 200:
    data = response.json()
    print(data)
    print(type(data))
else:
    print("http请求异常!")
