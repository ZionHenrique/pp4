from app import predict
import sys

img_path = r"Food Classification dataset/pizza/001.jpg"
print(f"Testing image: {img_path}")

# Compatibilidade: `predict` pode retornar (label, prob, extra_info)
result = predict(img_path)
label = None
prob = 0.0
extra = {}
if isinstance(result, tuple):
	if len(result) == 3:
		label, prob, extra = result
	elif len(result) == 2:
		label, prob = result
	elif len(result) == 1:
		label = result[0]
else:
	label = result

print(f"Prediction: {label}, probability: {prob}")
