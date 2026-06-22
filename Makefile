.PHONY: help data-small data-full preprocess train clean

help:
	@echo "Available targets:"
	@echo "  make data-small"
	@echo "  make data-full"
	@echo "  make preprocess"
	@echo "  make train"
	@echo "  make clean"

data-small:
	python3 scripts/download_data.py --mode small

data-full:
	python3 scripts/download_data.py --mode full

copy-data:
	cp -r /scratch/training/ecg_ltafdb/ltafdb ./data/raw/ltafdb

preprocess:
	python3 scripts/preprocess.py

train:
	python3 scripts/train.py

clean:
	rm -rf outputs/*

