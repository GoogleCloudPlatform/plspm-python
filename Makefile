init:
	pip3 install -r requirements.txt

test:
	mkdir -p test-results/pytest
	python3 -m pytest tests --junitxml=test-results/pytest/plspm_test_report.xml

.PHONY: init test
