# PFLD_LAB
I'm reading the paper of PFLD and do some experiment on it.
Based on PFLD offical codes(https://github.com/polarisZhao/PFLD-pytorch),I try to change some of the structure of the code to experiment.

### Changes based on offical codes
In offical codes, PFLD just calls MTCNN as face detector. I add yolov5 and pyramidbox so that we can choose face detector from these three.</br>
Additionally, I'm testing to replace the backbone of PFLD.</br>
You can use your own yolo models to have a better effient. The models are placed in `./FaceDetector/yolov5/weights/`</br>

### Setup
#### 1. Install requirements

~~~shell
pip3 install -r requirements.txt
~~~

#### 2. Datasets

- **WFLW Dataset Download**

​    [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing)  with 98 fully manual annotated landmarks.

1. WFLW Training and Testing images [[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)]
2. WFLW  [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)
3. Unzip above two packages and put them on `./data/WFLW/`
4. move `Mirror98.txt` to `WFLW/WFLW_annotations`

~~~shell
$ cd data 
$ python3 SetPreparation.py
~~~

#### 3. training & testing

training :

~~~shell
$ python3 train.py
~~~
use tensorboard, open a new terminal
~~~
$ tensorboard  --logdir=./checkpoint/tensorboard/
~~~
testing:

~~~shell
$ python3 test.py
~~~

#### 4. pytorch -> onnx -> ncnn

**Pytorch -> onnx**

~~~~shell
python3 pytorch2onnx.py
~~~~

**onnx -> ncnn**

how to build :https://github.com/Tencent/ncnn/wiki/how-to-build

~~~shell
cd ncnn/build/tools/onnx
./onnx2ncnn pfld-sim.onnx pfld-sim.param pfld-sim.bin
~~~

Now you can use **pfld-sim.param** and **pfld-sim.bin** in ncnn:

~~~cpp
ncnn::Net pfld;
pfld.load_param("path/to/pfld-sim.param");
pfld.load_model("path/to/pfld-sim.bin");

cv::Mat img = cv::imread(imagepath, 1);
ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 112, 112);
const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
in.substract_mean_normalize(0, norm_vals);

ncnn::Extractor ex = pfld.create_extractor();
ex.input("input_1", in);
ncnn::Mat out;
ex.extract("415", out);
~~~
