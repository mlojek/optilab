all_code = src tests

install_dependencies:
	pip install -r requirements.txt
	pip install -e .

install:
	pre-commit install

docker:
	docker build . -t mlojek/optilab:15

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
