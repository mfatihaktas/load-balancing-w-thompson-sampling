PYTEST=pytest --color=yes --verbose --showlocals
PROFILE_FILE="profile.dump"


clean:
	rm -fr .direnv

env:
	pyenv install; \
	direnv allow

install:
	pip install --upgrade pip; \
	pip install poetry; \
	poetry install; \
	pip install -e .

profile:
	# python tests/test_optimal_vs_ts.py
	python -m cProfile -o ${PROFILE_FILE} tests/test_optimal_vs_ts.py

viz:
	snakeviz ${PROFILE_FILE}

test:
	${PYTEST} "tests/test_optimal_vs_ts.py::test_optimal_vs_ts"
