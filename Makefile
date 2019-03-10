init:
	python3 -m pip install --user -r requirements.txt

test:
	mkdir -p test-results/pytest
	python3 -m pytest tests --disable-warnings --junitxml=test-results/pytest/plspm_test_report.xml

package:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

docs:
	sphinx-apidoc -o docs/source plspm
	cd docs && sphinx-build -M html . .

.PHONY: init test docs
