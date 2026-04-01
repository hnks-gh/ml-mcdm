#!/usr/bin/env python3
"""Clean up duplicated Discussion section"""

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: after "governance reform." there's a broken "  The three innovations..." section
# followed by a duplicate "Despite the methodological rigor" and full duplicate "In conclusion..."

# Simple approach: find the broken part and remove it
broken_marker = "governance  The three innovations"
if broken_marker in content:
    # Find where this broken part starts
    idx = content.find(broken_marker)
    if idx > 0:
        # Go back to find "governance " to replace it properly
        start_idx = content.rfind("governance", idx - 100, idx) + len("governance")
        
        # Find the end - look for the second occurrence of the conclusion paragraph
        end_idx = content.find("In conclusion, this research presents a methodological paradigm shift", idx + 1)
        
        if end_idx > 0:
            # Replace everything from " The three..." to just before the duplicate "In conclusion..."
            # with just " reform.\n\n"
            content_before = content[:start_idx]
            content_after = "\nrepair.\n\n" + content[end_idx:]
            
            # Oops, let me be more precise
            # We want to replace from the extra space onwards
            extra_text_start = idx - len("governance")
            # Find the correct end - it's before "In conclusion" the second time
            second_conclusion_idx = content.find("In conclusion, this research", idx)
            
            if second_conclusion_idx > end_idx:
                second_conclusion_idx = end_idx
                
            content = content[:extra_text_start + len("governance")] + " reform.\n\n" + content[second_conclusion_idx:]

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("File fixed")
