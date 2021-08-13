# 导入工具包
import paddlehub as hub
import numpy as np
'''
Authors' units: Computer Vision group by RobAI-Lab, Hainan university
Author：Nemo-Lv, Hui-Wang, Jiangpeng-Li, Haojiang-Zhao

环境需求(Nemo-Lv)
boost: 0.1
cmake: 3.21.0
dlib: 19.22.0
opencv-python: 4.5.3.56
paddle: 2.1(官网安装合适框架即可，文档说1.8以上即可，实测2.1才行)(建议单独创建虚拟环境，以防止框架之间相互污染环境)
paddlehub: 2.1.0
scipy: 1.7.0
'''
# 模型载入
module = hub.Module(name="pyramidbox_lite_server_mask")

# 判断人脸
def pyramidbox_detect(frame):
    results = module.face_detection(images=[frame])
    if results[0]['data']:
        # print(results)
        data = results[0]['data'][0]
        # print(results[0]['data'][0]['top'])   断点
        x1 = int(data['left'])
        y1 = int(data['top'])
        x2 = int(data['right'])
        y2 = int(data['bottom'])
        ret = np.array([[x1, y1, x2, y2]])
        return ret, 0
    else:
        return np.array([[0, 0, 0, 0]]), 0