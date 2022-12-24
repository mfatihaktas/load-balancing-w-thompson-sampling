PYTEST=pytest --color=yes --verbose --showlocals

test:
	${PYTEST} "tests/test_optimal_vs_ts.py::test_optimal_vs_ts"
