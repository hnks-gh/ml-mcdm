#!/usr/bin/env python3
"""Fix the duplication in the Discussion section"""

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find section boundaries
discussion_start = None
conclusion_start = None

for i, line in enumerate(lines):
    if r'\section{Discussion}' in line:
        discussion_start = i
    if r'\section{Conclusion}' in line:
        conclusion_start = i
        break

if not discussion_start or not conclusion_start:
    print("Could not find Discussion or Conclusion sections")
    exit(1)

print(f"Discussion starts at line {discussion_start+1}")
print(f"Conclusion starts at line {conclusion_start+1}")

# Find the duplicate content
# Strategy: Find the first "In conclusion" that's NOT the last one before Conclusion section
in_conclusion_indices = []
for i in range(discussion_start, conclusion_start):
    if 'In conclusion' in lines[i]:
        in_conclusion_indices.append(i)

print(f"Found '{len(in_conclusion_indices)}' occurrences of 'In conclusion'")
print(f"At lines: {[i+1 for i in in_conclusion_indices]}")

if len(in_conclusion_indices) > 1:
    # Keep only the last "In conclusion" paragraph
    # The issue is that after the first one, there should be a limitspan paragraph for limitations
    # and then the final one. But we have duplicates.
    
    # Look at what's between the first and second "In conclusion"
    first_conclusion_line = in_conclusion_indices[0]
    second_conclusion_line = in_conclusion_indices[1] if len(in_conclusion_indices) > 1 else None
    
    print(f"\nFirst 'In conclusion' at line {first_conclusion_line + 1}")
    if second_conclusion_line:
        print(f"Second 'In conclusion' at line {second_conclusion_line + 1}")
        
        # Check if there's duplicate "Despite the methodological rigor"
        despite_count = 0
        despite_indices = []
        for i in range(discussion_start, conclusion_start):
            if 'Despite the methodological rigor' in lines[i]:
                despite_count += 1
                despite_indices.append(i)
        
        print(f"Found '{despite_count}' occurrences of 'Despite the methodological rigor'")
        print(f"At lines: {[i+1 for i in despite_indices]}")
        
        if despite_count > 1:
            # Remove everything from the end of the first "In conclusion" paragraph
            # until the start of the second "In conclusion" paragraph
            # This includes the broken sentence " The three innovations..." and the duplicate "Despite..."
            
            # Find where the first "In conclusion" paragraph ends (look for the sentence ending after "reform.")
            search_text = "establishing a foundation for evidence-based governance reform."
            
            # Find the first occurrence after line 1820 (approximate)
            first_reform_line = None
            for i in range(first_conclusion_line, second_conclusion_line):
                if "establishing a foundation for evidence-based governance reform." in lines[i]:
                    first_reform_line = i
                    break
            
            print(f"\nFirst occurrence of 'reform.' at line {first_reform_line + 1}")
            
            if first_reform_line:
                # Delete everything after this line until the Conclusion section
                # But keep the % ===== Conclusion header
                
                # Find the line with \section{Conclusion}
                conclusion_line_idx = None
                for i in range(first_reform_line, conclusion_start + 5):
                    if r'\section{Conclusion}' in lines[i]:
                        conclusion_line_idx = i
                        break
                
                if conclusion_line_idx:
                    print(f"Conclusion header at line {conclusion_line_idx + 1}")
                    
                    # Keep lines up to and including the first reform sentence
                    # Then add two blank lines and jump to the Conclusion section
                    lines_to_keep = lines[:first_reform_line + 1]
                    
                    # Add newlines before Conclusion
                    lines_to_keep.append("\n")
                    lines_to_keep.append("% =============================================================\n")
                    
                    # Add the Conclusion section and everything after
                    lines_to_keep.extend(lines[conclusion_line_idx:])
                    
                    # Write back
                    with open('paper/main.tex', 'w', encoding='utf-8') as f:
                        f.writelines(lines_to_keep)
                    
                    print("\nFile fixed!  Removed duplicate content.")
                else:
                    print("ERROR: Could not find Conclusion section")
            else:
                print("ERROR: Could not find 'reform.' sentence")
        else:
            print("No duplicate 'Despite' sections found")
    else:
        print("Only one occurrence of 'In conclusion' found")
else:
    print("Only one occurrence of 'In conclusion' found")
