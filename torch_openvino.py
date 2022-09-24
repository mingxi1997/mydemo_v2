import torchvision
import torch
model=torchvision.models.alexnet()
model.eval()


height, width = 224, 224
dummy_input = torch.randn(1, 3, height, width)

output_path = 'tt.onnx'
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    verbose=False,
    keep_initializers_as_inputs=True,
    opset_version=11,
)


import cv2
import onnxruntime
import numpy as np

onnx_model_path = 'tt.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path,providers=['TensorrtExecutionProvider'])
image = cv2.imread('/home/xu/Pictures/11.jpeg')  # load the test image
# 使用你的算法前处理部分 同时前处理输出必须为NCHW格式的numpy数组
image=cv2.imread('/home/xu/Pictures/test.jpg')
image=cv2.resize(image, (224,224))

image = image.transpose(2,0,1).reshape(1,3,224,224).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: image}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_path='tt.onnx'
mo_command = f"""mo
                  --input_model "{onnx_path}"
                  --input_shape "[1,3, 224,224]"
                  --mean_values="[123.675, 116.28 , 103.53]"
                  --scale_values="[58.395, 57.12 , 57.375]"
                  --data_type FP16
                  --output_dir "model"
                  """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")





from openvino.runtime import Core

ie = Core()
classification_model_xml = "/home/xu/utest/model/tt.xml"
model = ie.read_model(model=classification_model_xml)
input_layer = model.input(0)
output_layer = model.output(0)

print(f"input precision: {input_layer.element_type}")
print(f"input shape: {input_layer.shape}")
print(f"output precision: {output_layer.element_type}")
print(f"output shape: {output_layer.shape}")




compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

import cv2
import numpy as np 
image=cv2.imread('/home/xu/Pictures/test.jpg')
image=cv2.resize(image, (224,224))

image = image.transpose(2,0,1).reshape(1,3,224,224).astype(np.float32)
result = compiled_model([image])[output_layer]



















