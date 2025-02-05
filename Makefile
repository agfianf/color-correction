yolo-export-onnx:
	yolo export \
	model=color_correction/asset/.model/yv8-det.pt \
	format=onnx \
	device=mps \
	simplify=True \
	dynamic=False \
	half=True

test:
	pytest tests -v


diff:
	git diff main..{branch_name} > diff-output.txt

log:
	git log --oneline main..{branch_name} > log-output.txt

update-uv-lock:
	uv lock
