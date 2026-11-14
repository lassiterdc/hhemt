```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install --no-deps "swmm-toolkit==0.15.5" "pyswmm==2.1.0" "swmmio==0.8.2"
pip install -e . --no-deps
```
