.PHONY: data features eda train tune test all

PY ?= python

data:
	$(PY) -m src.data.download_flights
	$(PY) -m src.data.download_weather

features:
	$(PY) -m src.preprocessing
	$(PY) -m src.build_features

eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train:
	$(PY) -m src.train
	$(PY) -m src.train --prediction-point pre_departure

tune:
	$(PY) -m src.tune
	$(PY) -m src.tune --prediction-point pre_departure

test:
	$(PY) -m pytest -q tests

all: data features train tune
