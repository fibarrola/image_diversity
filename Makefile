pretty:
	black --config black.toml .

lint:
	black --config black.toml --check .
	flake8 --config flake8.cfg .