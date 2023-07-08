# Use double precision

from onnx_opt import convert_onnx_to_double
onnx_model = convert_onnx_to_double(path)
dummy = torch.randn([1, *onnx_shape], dtype=torch.float64)
pytorch_model.to(torch.float64)
output_pytorch = pytorch_model(dummy).numpy()
sess = ort.InferenceSession(onnx_model.SerializeToString())
output_onnx = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
