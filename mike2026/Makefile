.PHONY: analysis validate assets test
analysis:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_revision3_analysis.py
validate:
	python scripts/validate_revision3.py
assets:
	python scripts/make_revision3_assets.py
test:
	python -m unittest discover -s tests -v
