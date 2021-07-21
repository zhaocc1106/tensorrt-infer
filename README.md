# tensorrt-infer
TensorRT模型加载与推理

## 环境
Cuda10.0 + Cudnn7.6 + TensorRT7.0.0.11

## 运行
先通过TensorRT中的trtexec工具转换onnx模型到trt：<br>
``
trtexec --onnx=./model.onnx --explicitBatch --minShapes=inp0:1x28x28x3 --optShapes=inp0:64x28x28x3 --maxShapes=inp0:64x28x28x3 --shapes=inp0:64x28x28x3 --saveEngine=./model.trt
``<br>
使用生成model.trt来推理：<br>
``
./trt_test model.trt
``