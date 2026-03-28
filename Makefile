.PHONY: help install test export benchmark clean

# Default target: show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Install dependencies (Torch, NumPy, ONNX Runtime)"
	@echo "  test       Run mathematical consistency tests (Torch vs NumPy vs ONNX)"
	@echo "  export     Export Mamba step function to ONNX (mamba_step_inference.onnx)"
	@echo "  benchmark  Run performance benchmark across implementations"
	@echo "  clean      Remove temporary files and exported models"

install:
	pip install .
	pip install torch onnx onnxruntime tabulate

test:
	PYTHONPATH=. python3 tests/test_step_consistency.py

export:
	PYTHONPATH=. python3 scripts/export_onnx_step.py

benchmark: export
	PYTHONPATH=. python3 scripts/benchmark.py

clean:
	rm -f mamba_step_inference.onnx
	rm -f benchmark_results.md
	rm -f tests/temp_consistency.onnx
	rm -rf __pycache__
	rm -rf */__pycache__
