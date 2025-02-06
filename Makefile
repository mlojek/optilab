all_code = src tests

install:
	pip install -e .

setup: install
	pre-commit install

docker: clean
	docker build . -t mlojek/optilab

clean:
	git clean -fdx

format:
	isort ${all_code} --profile black
	black ${all_code}

check: format
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code}

test:
	pytest

doc:
	sphinx-apidoc -o docs src/optilab -f
