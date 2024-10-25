clean:
	rm -rf build
	rm -rf src/sofes.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

format:
	isort **/*.py
	black **/*.py

check:
	# mypy src sandbox
	# flake8
	black **/*.py --check
	isort **/*.py --check
	pylint src sandbox tests

test:
	pytest
