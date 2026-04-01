import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove the broken part: replace 'governance  The three...' with just 'governance reform.'
pattern = r'governance\s+The three innovations.*?(?=\n\nDespite the methodological rigor of the proposed hybrid framework, certain limitations remain that provide fertile ground for future scholarly inquiry\. A primary constraint)'

text = re.sub(pattern, 'governance reform.', text, flags=re.DOTALL)

# Now remove the duplicate "Despite...In conclusion" section
# Find first "Despite...Subsequent studies" 
first_despite_pos = text.find('Despite the methodological rigor of the proposed hybrid framework')
if first_despite_pos >= 0:
    second_despite_pos = text.find('Despite the methodological rigor of the proposed hybrid framework', first_despite_pos + 100)
    if second_despite_pos > first_despite_pos:
        # Delete the second one and everything until the next "\section{Conclusion}"
        section_conclusion_pos = text.find(r'\section{Conclusion}', second_despite_pos)
        if section_conclusion_pos > second_despite_pos:
            # Find where the first "Despite" section ends (look for the limits paragraph end)
            first_despite_end = text.find('Subsequent studies should also prioritize the development of ``cross-walk``', first_despite_pos)
            first_despite_end = text.find('maintaining the continuity of the PAPI time-series analysis.', first_despite_end)
            if first_despite_end > 0:
                first_despite_end = text.find('\n\n', first_despite_end)
                # Delete everything from end of first Despite through start of Conclusion
                #text text = text[:first_despite_end] + '\n\n' + text[section_conclusion_pos:]
                
                # Find the start of "In conclusion" that comes after the second Despite
                in_conclusion_pos = text.find('In conclusion, this research presents', second_despite_pos)
                if in_conclusion_pos > 0:
                    # Delete from right after the proper limitations paragraph to right before \section{Conclusion}
                    text = text[:first_despite_end] + '\n\n' + text[section_conclusion_pos:]

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('Fixed!')
