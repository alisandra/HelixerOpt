# Data Wrangling

OK, in this larger project we're bringing in data from three "bioinformatics formats". 
All of which generally cover a whole genome.


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

## counting
Throughout we'll be counting from 1 and with inclusive ends for all intermediate
formats, because it's
common practice throughout most bioinformatics "formats", and battles must
be chosen wisely.

## 1:1 pre-processing
After this step, everything that is shared between the the formats 
(coordinates, sequence, version, species) should be stored with the same spec. 
For development, all files will be .json, but this will probably be replaced with
something smaller, at least for the coverage data.

- bam -> coverage.json
- gff -> annotation.json
- fasta -> sequence.json

### sequence.json
#### genome level
- species: string
- accession: string
- version: string
- acquired_from: string
- genome_size: int
- gc_content: 0 <= int <= genome_size
- cannonical_kmer_content: {'aaa': 0 <= int <= genome_size - 2, 'aac': ...}
  - sum of values = genome_size - 2
- ambiguous_content: 0 <= int <= genome_size

#### sequence level
- seqid: string
- set: None, but theoretically one of [test, train, dev, None]
- type: one of [chromosome, plasmid, scaffold, contig, unspecified]
- start: 1 
- end: int (here, the sequence length)
- length: int (end - start + 1)
- gc_content: 0 <= int <= genome_size
- cannonical_kmer_content: {'aaa': 0 <= int <= genome_size - 2, 'aac': ...}
  - sum of values = genome_size - 2
- ambiguous_content: 0 <= int <= genome_size
- sequence: string

### annotation.json
#### genome level
- species: string
- accession: string
- version: string
- acquired_from: string

#### sequence level 
- seqid: string
- set: None, but theoretically one of [test, train, dev, None]
- start: 1 
- end: int (here, the sequence length)
- length: int (end - start + 1)
- strand: +, but theoretically one of [+, -]
- annotations: hierarchical 'feature storage'
  - id
  - source: string
  - type: one of [intergenic, erroneous, gene, mRNA, exon, CDS, intron, UTR, TSS, TTS, tRNA, 
  rRNA, ncRNA, ...]
  - start: int
  - end: int
  - phase: int
  - score: float
  - complete: boolean
  - <flat attribute or all supported keys, or just make a dictionary out of the attribute?>
  - children:
  
  A coding gene must always be made up of the following:
```
  gene{
    mRNA[1+][children by ID:TSS,exon,(intron,exon)[0+],TTS]
    TSS
    exon{ 
         UTR[0-1], CDS[0-1]
         }
    intron
    TTS
    }
  }
```
Note that if an exon has a different combination of UTR/CDS when in a different
transcript (e.g. an exon that is sometimes 1st, sometimes 2nd that includes a 
possible start codon); it must be included independently for each case.

This format is not compatible with trans-splicing and any such transcripts will
be broken into pieces and treated as independent partials (or marked as erroneous).
<or maybe I can add a 'distant-child' field to genes or something?...>

Every bp of the sequence should be covered, so that nothing is stored _implicitly_.
That means intergenic, and intronic regions are marked. Of course, there is still
the option of having multiple overlapping transcripts or genes. 
 
Where original gffs could not be parsed -> _erroneous_ type.

### coverage.json
#### genome level
- species: string
- accession: string
- version: string
- acquired_from: string
#### sequence level 
- seqid: string
- set: None, but one of [test, train, dev, None]
- start: 1 
- end: int (here, the sequence length)
- length: int (end - start + 1)
- coverage_runs:
  - local_start: [ints for each position where a run (with identical coverage) starts]
  - coverage: [ints for coverage of run]
  - spliced_coverage: [ints for coverage of reads that are spliced across this run]

## Splitting

- generic function to setup [seqid, start, end, set] for every location.
- custom function for each of the above data types to split, set the 4 fields and...
  - sequence.json: parse sub-sequence
  - annotation.json: distribute annotations by region, handle splitting of features
  and mark things as 'not complete' as necessary
  - coverage.json: distribute coverage by region, recalculate 'local_start'

## TFRecord creation

This is the usual end result of `--generate_data`, problem-specific, sharded,
TFRecord files, with examples fully processed for feeding into a network.