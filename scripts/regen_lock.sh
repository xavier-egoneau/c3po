#!/bin/bash
set -e
python3 -m pip freeze --user > requirements-lock.txt
git add requirements-lock.txt
