# Data Wrangling

OK, in this larger project we're bringing in data from three "bioinformatics formats". 
All of which are generally cover a whole genome.


- BAM: read alignments / expression data, large and needs preprocessing
- gff: parsing throws way too many errors, too much is stored implicitly
- fasta: would probably be ok, but also wouldn't mind more structured 
  meta-data than fasta headers
  
Therefore, will be doing what I don't much want to be doing, and making more
formats. Well, specs for JSON format (and maybe also smaller / faster versions later).

Data will be processed in a multi tiered manner:

- 1:1 pre-processing (a file in, a cleaned and possibly reformatted file out)
- assignment to dev/train/test and splitting of large sequences
- TFRecord creation


