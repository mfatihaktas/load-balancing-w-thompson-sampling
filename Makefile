PYTEST=pytest --color=yes --verbose --showlocals

clean:
	rm -fr .direnv

env:
	direnv allow

install:
	pyenv install; \
	direnv allow; \
	pip install --upgrade pip; \
	pip install poetry; \
	poetry install; \
	pip install -e .

test:
	${PYTEST} "tests/test_optimal_vs_ts.py::test_optimal_vs_ts"
