#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = diabetes_classif
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format







#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) dm_classif/dataset.py

dataset:
	$(PYTHON_INTERPRETER) dm_classif/dataset.py

preprocess: 
	$(PYTHON_INTERPRETER) dm_classif/features.py

train:
	$(PYTHON_INTERPRETER) dm_classif/modeling/train.py

train-recall:
	$(PYTHON_INTERPRETER) dm_classif/modeling/train.py --metric recall

train-accuracy:
	$(PYTHON_INTERPRETER) dm_classif/modeling/train.py --metric accuracy

train-precision:
	$(PYTHON_INTERPRETER) dm_classif/modeling/train.py --metric precision

full: dataset preprocess train-recall

.PHONY: predict
predict:
	@echo "Running model inference and evaluation..."
	python -m dm_classif.modeling.predict

## Full pipeline (ingestion → preprocessing → training with accuracy)
.PHONY: full-accuracy
full-accuracy: dataset preprocess train-accuracy

## Full pipeline (ingestion → preprocessing → training with precision)
.PHONY: full-precision
full-precision: dataset preprocess train-precision

## Full pipeline (ingestion → preprocessing → training with recall)
.PHONY: full-recall
full-recall: dataset preprocess train-recall

test: 
	$(PYTHON_INTERPRETER) dm_classif/modeling/predict.py

plots:
	$(PYTHON_INTERPRETER) dm_classif/plots.py


## Clean key generated outputs only (pipeline files, model artifacts, and reports)
.PHONY: clean-artifacts
clean-artifacts:
	@echo "Cleaning processed pipeline files and model artifacts..."
	$(PYTHON_INTERPRETER) scripts/clean_artifacts.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
