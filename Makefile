yolo-export-onnx:
	yolo export \
	model=color_correction_asdfghjkl/asset/.model/yv8-det.pt \
	format=onnx \
	device=mps \
	simplify=True \
	dynamic=False \
	half=True

test:
	pytest tests -v