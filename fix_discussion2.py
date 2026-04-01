#!/usr/bin/env python3
"""Fix the duplication in the Discussion section"""

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the discussion section
disc_idx = content.find(r'\section{Discussion}')
conc_idx = content.find(r'\section{Conclusion}')

if disc_idx < 0 or conc_idx < 0:
    print("Could not find sections")
    exit(1)

# Extract the discussion content
discussion_content = content[disc_idx:conc_idx]

# Find all occurrences of "In conclusion"
import re
matches = [(m.start() + disc_idx, m.group()) for m in re.finditer(r'In conclusion', content[disc_idx:conc_idx])]

print(f"Found {len(matches)} 'In conclusion' occurrences")

# Find the broken "  The three innovations" part that shouldn't be there
bad_section_start = content.find("governance  The three innovations", disc_idx, conc_idx)

if bad_section_start > 0:
    print(f"Found broken section at position {bad_section_start}")
    
    # Find the end of this broken section (looking for "Despite the methodological rigor")
    bad_section_end = content.find("Despite the methodological rigor", bad_section_start, conc_idx)
    
    if bad_section_end > 0:
        # Find again to get past this duplicate "Despite"
        bad_section_end = content.find("Subsequent studies", bad_section_end, conc_idx)
        if bad_section_end > 0:
            # Find the rest to delete
            bad_section_end = content.find("\n\nIn conclusion", bad_section_end, conc_idx)
            if bad_section_end > 0:
                # Also need to delete the duplicate "In conclusion" and everything up to the second one
                bad_section_end = content.find("establishing a foundation for evidence-based governance reform.", bad_section_end, conc_idx)
                bad_section_end = content.find(".", bad_section_end)  # Find the period after reform
                
                if bad_section_end > 0:
                    # Now delete from "governance" to this period
                    delete_start = content.rfind("governance", disc_idx, bad_section_start) + len("governance")
                    delete_end = bad_section_end + 1
                    
                    #Actually, let's be more specific
                    # We want to replace "governance  The three..." with "governance reform."
                    # Then remove the duplicate "Despite..." and second "In conclusion..."
                    
                    fixed_content = content[:bad_section_start] + "governance reform." + content[bad_section_end + 2 :]
                    
                    # Now check if there are still duplicates
                    if fixed_content.count("Despite the methodological rigor") > 1:
                        print("Still have duplicates, need further fix")
                        # Find and remove the second "Despite...Subsequent studies..." section
                        first_despite = fixed_content.find("Despite the methodological rigor", disc_idx)
                        second_despite = fixed_content.find("Despite the methodological rigor", first_despite + 100)
                        
                        if second_despite > 0:
                            # Find where this duplicate ends (before \section{Conclusion})
                            second_despite_end = fixed_content.find("establishing a foundation for evidence-based governance reform.", second_despite)
                            second_despite_end = fixed_content.find(".  ", second_despite_end) + 1
                            
                            # Delete the duplicate despite section
                            fixed_content = fixed_content[:first_despite + len("Despite...")] + "\n\n" + fixed_content[second_despite_end + 2:]
                    
                    with open('paper/main.tex', 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    print("File fixed!")
                else:
                    print("Could not find reform period")
            else:
                print("Could not find conclusion point")
        else:
            print("Could not find subsequent studies")
    else:
        print("Could not find despite point")
else:
    print("Could not find the broken section with 'The three innovations'")
