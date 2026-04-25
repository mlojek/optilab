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
	ruff format ${all_code}
	pyprojectsort pyproject.toml

check: format
	ruff check ${all_code}
	pyprojectsort pyproject.toml --check

test:
	pytest

doc:
	sphinx-apidoc -o docs src/optilab -f
