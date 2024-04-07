
PYPI_TOKEN := $(shell cat ~/.pypi_token.txt)

default:
	python setup.py install

test:
	python -m pytest --envfile env

push_to_pypi:
	rm -fr dist
	python3 -m build
	twine upload -r pypi dist/* --user __token__ --password $(PYPI_TOKEN) --skip-existing
	rm -fr dist

github_actions:
	act