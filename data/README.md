# Data

The admissions dataset (`TU.csv`) is confidential, student-level data provided by Trinity University for coursework, and is not included in this repository.

With authorized access, place `TU.csv` in this folder. The notebook reads `../data/TU.csv` and writes the cleaned training set to `../data/cleaneddftrain.csv`. Both files are git-ignored.

**Shape:** 15,143 admitted first-year applicants (10,000 train / 5,143 test, predefined), 69 columns. Entry terms Fall 2017 through Fall 2021. The target is `Decision` (1 = enrolled).
