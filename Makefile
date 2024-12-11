all_code = src sandbox tests

install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

clean:
	rm -rf build
	rm -rf src/optilab.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

format:
	isort ${all_code} --profile black
	black ${all_code}

check:
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code}

test:
	pytest

doc:
	sphinx-apidoc -o docs src/optilab
