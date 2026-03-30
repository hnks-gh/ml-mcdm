import re

def main():
    with open(r'c:\Users\hoang\Documents\ml-mcdm\forecasting\unified.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = []
    for i, line in enumerate(lines, 1):
        if 'def stage6' in line or 'def _inverse_' in line or 'def stage5' in line or 'stage6_' in line or 'def _evaluate' in line:
            out.append(f"Line {i}: {line.strip()}")
            
    with open(r'c:\Users\hoang\Documents\ml-mcdm\find_lines_output.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))

if __name__ == '__main__':
    main()
