all_code = src tests

install:
	pip install -e .

install_dev: install
	pre-commit install

build_wheel:
	pip install build wheel twine
	python -m build --wheel . --outdir dist/

docker: clean
	docker build . -t mlojek/optilab

clean:
	git clean -fdx

format:
	isort ${all_code} --profile black
	black ${all_code}
	pyprojectsort pyproject.toml

check: format
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code}
	pyprojectsort pyproject.toml --check

test:
	pytest

doc:
	sphinx-apidoc -o docs src/optilab -f
