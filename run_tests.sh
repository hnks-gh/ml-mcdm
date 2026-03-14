#!/bin/bash
cd /c/Users/hoang/Documents/ml-mcdm
python -m pytest tests/ --tb=short -q > /c/Users/hoang/Documents/ml-mcdm/test_output.txt 2>&1
echo "Exit code: $?" >> /c/Users/hoang/Documents/ml-mcdm/test_output.txt
