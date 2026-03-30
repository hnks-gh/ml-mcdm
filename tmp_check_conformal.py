import sys
def check():
    with open(r'c:\Users\hoang\Documents\ml-mcdm\forecasting\conformal.py', 'r', encoding='utf-8') as f:
        src = f.readlines()
    
    out = []
    for i, line in enumerate(src, 1):
        if 'def calibrate_residuals' in line or 'def _compute_q_hat' in line or 'q_hat =' in line or 'def calibrate(' in line:
            out.append(f"Line {i}: {line.strip()}")
            
    with open(r'c:\Users\hoang\Documents\ml-mcdm\conformal_check.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
check()
