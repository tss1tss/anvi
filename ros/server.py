from flask import Flask, request
import signal
import rospy
from sensor_msgs.msg import CompressedImage
from cira_msgs.srv import CiraFlowService, CiraFlowServiceRequest, CiraFlowServiceResponse
import json
import base64
def handler(signum, frame):
    exit(1)

signal.signal(signal.SIGINT, handler)


ros_node_name = "api_node"
serviceName = "api_ros"

app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}


@app.route('/')
def index():
    return {'success': True}

@app.route('/ros', methods=['POST', 'GET'])
def func1():
    if request.is_json:
        param = request.json
        print(param)

    req = CiraFlowServiceRequest()

    jsonstr = request.form.get('jsonstr', '{}')
    req.flow_in.jsonstr = jsonstr

    # jsonstr = {'a': 1}
    # req.flow_in.jsonstr = json.dumps(jsonstr)

    if 'image' in request.files:
        cm = CompressedImage()
        image = request.files['image']
        cm.data = image.stream.read()
        req.flow_in.img = cm

    try:
        res: CiraFlowServiceResponse = ciraService.call(req)
        print(res.flow_out.jsonstr)
        
        jsonstrRes = json.loads(res.flow_out.jsonstr)
        print(jsonstrRes)
        print(json.dumps(jsonstrRes, indent=2))

        print(len(res.flow_out.img.data))

        jsoRes = {
            'success': True,
            'payload': jsonstrRes['payload'],
            'img': {
                'data': base64.b64encode(res.flow_out.img.data).decode(),
                'format': res.flow_out.img.format,
            }
        }

        return jsoRes
    except:
        return {'success': False}


if __name__ == "__main__":
    rospy.init_node(ros_node_name, anonymous=True)
    print(f"wait for service [{serviceName}]")
    rospy.wait_for_service(serviceName)
    ciraService = rospy.ServiceProxy(serviceName, CiraFlowService)
    app.run(host='0.0.0.0', port=3000)
    