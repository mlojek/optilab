all_code = src tests

install:
	uv sync

install_dev: install
	uvx pre-commit install

build_wheel:
	uv build

docker: clean
	docker build . -t mlojek/optilab

clean:
	git clean -fdx

format:
	uvx ruff format ${all_code}
	uvx pyprojectsort pyproject.toml

check: format
	uvx ruff check ${all_code}
	uvx ty check ${all_code}
	uvx pyprojectsort pyproject.toml --check

test:
	uv run pytest

doc:
	uvx sphinx-apidoc -o docs src/optilab -f
