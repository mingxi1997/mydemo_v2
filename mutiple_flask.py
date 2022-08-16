from flask import Flask
from threading import Thread
import os

app1 = Flask('app1')

@app1.route('/')
def foo():
    return '1'

Thread(target=lambda: app1.run(port=5001)).start()

# ----------服务2-----------------

app2 = Flask('app2')

@app2.route('/')
def bar():
    return 'hello world'

app2.run(debug=True, port=5002)

if __name__ == '__main__':
    os.environ["WERKZEUG_RUN_MAIN"] = 'true'
    Thread(target=app1).start()
    app2()


