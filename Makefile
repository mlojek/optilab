all_code = src sandbox tests

clean:
	rm -rf build
	rm -rf src/sofes.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

format:
	isort ${all_code}
	black ${all_code}

check:
	black ${all_code} --check
	isort ${all_code} --check
	pylint ${all_code}

test:
	pytest
