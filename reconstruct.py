#!/usr/bin/env python3
"""Reconstruct the Discussion section properly"""

import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find Discussion and Conclusion sections
discussion_pattern = r'(\\section\{Discussion\}\s*\\label\{sec:discussion\}\s*% =.*?\n)'
conclusion_pattern =r'(\\section\{Conclusion\}\s*\\label\{sec:conclusion\})'

disc_match = re.search(discussion_pattern, content, re.DOTALL)
conc_match = re.search(conclusion_pattern, content)

if not disc_match or not conc_match:
    print("Could not find sections")
    exit(1)

discussion_header_end = disc_match.end()
conclusion_start = conc_match.start()

discussion_text = content[discussion_header_end:conclusion_start]

# Count key phrases
print("Discussion section stats:")
print(f"  'In conclusion' count: {discussion_text.count('In conclusion')}")
print(f"  'Despite the methodological rigor' count: {discussion_text.count('Despite the methodological rigor')}")

# The structure should be:
# Paragraph 1: main findings
# Paragraph 2: theoretical/practical implications
# Paragraph 3: limitations
# Paragraph 4: concluding statement
# 
# But we have duplicates of paragraphs 3 and 4

# Find where the first "In conclusion" paragraph ends (should end with "reform.\n\n")
lines = discussion_text.split('\n\n')  # Paragraphs separated by double newline

# Reconstruct: keep only the right paragraphs
para1_end = discussion_text.find('These findings carry substantial theoretical')
para2_end = discussion_text.find('Despite the methodological rigor of the proposed hybrid framework')
para3_end = discussion_text.find('Subsequent studies should also prioritize')
para3_end = discussion_text.find('\n\n', para3_end)

# Extract paragraphs
para1 = discussion_text[:para1_end].strip() + '\n\n'
para2 = discussion_text[para1_end:para2_end].strip() + '\n\n'
para3 =discussion_text[para2_end:para3_end].strip() + '\n\n'

# Now find the proper conclusion paragraph (after all the limitations)
# It should be the one starting with "In conclusion, this research presents a methodological paradigm shift"
conc_start_within = discussion_text.find('In conclusion, this research presents a methodological paradigm shift', para3_end)
if conc_start_within < 0:
    # Try different text
    conc_start_within = discussion_text.rfind('In conclusion,', 0, conclusion_start - discussion_header_end)

if conc_start_within >= 0:
    para4 = discussion_text[conc_start_within:].strip()
else:
    para4 = ''

# Reconstruct the discussion
new_discussion = para1 + para2 + para3 + '\n\n' + para4.rstrip() + '\n\n'

# Replace in content
new_content = content[:discussion_header_end] + new_discussion + content[conclusion_start:]

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("File reconstructed successfully")
