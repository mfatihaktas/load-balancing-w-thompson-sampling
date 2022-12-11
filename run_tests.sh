#!/bin/bash


PYTEST="pytest --color=yes --verbose --showlocals"

${PYTEST} 'tests/test_ts_sliding_win.py::test_ThompsonSampling_slidingWin_vs_slidingWinForEachNode'
