grep 'pdb' EEH_clus4_331.txt >> all_pdbs.txt
grep 'pdb' EEH_clus4_331.txt -v >> eeh_new.txt
grep 'Cen' eeh_new.txt -v >> eeh_newer2.txt