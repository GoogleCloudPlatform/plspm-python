init:
	pip3 install -r requirements.txt

test:
	mkdir -p test-results/pytest
	python3 -m pytest tests --disable-warnings --junitxml=test-results/pytest/plspm_test_report.xml

package:
	python setup.py sdist bdist_wheel # 'python3 -m twine upload dist/*' to upload

.PHONY: init test
