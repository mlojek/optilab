clean:
	rm -rf build
	rm -rf src/sofes.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

format:
	isort .
	black .

check-src:
	echo mypy
	echo flake8
	echo black
	echo isort
	pylint src

check-sandbox:
	echo mypy
	echo flake8
	echo black
	echo isort
	pylint sandbox

test:
	echo pytest
