"""
Utility script to locate specific method definitions within large source files.

Used primarily for development diagnostics and cross-referencing line 
numbers in the UnifiedForecaster module.
"""

import sys

def find_lines() -> None:
    """
    Search for predefined target strings in unified.py and print their line numbers.
    """
    target_strings = ['def stage6', 'def _inverse_', 'def stage5']
    with open(r'c:\Users\hoang\Documents\ml-mcdm\forecasting\unified.py', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if any(ts in line for ts in target_strings):
                print(f"Match found at line {i}: {line.strip()}")

if __name__ == '__main__':
    find_lines()
