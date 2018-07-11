# gffreader

functions and classes to assist in the parsing and interpreting of gff files

## Functional bits

### Line oriented

#### GFFLine

Main class for taking a gffentry. It is defined from a line of a gfffile, basically
names all the classes. Makes numbers numbers, and similar. Has two additional 
attributes: .comment for commented line and .children for possible hierarchical ordering
of gffentries.

Has some fluff for swapping orientation, adding children, exporting gffline, etc, but not much.

Can enter itself onto an interval tree (although more interesting for subclasses).

In anycase the data for interval tree is setup through the MinimalData class for consistency
checking. But then passed as a tuple.

#### GFFLineMrna(GFFLine)

To be used for a gffline known to be mRNA/protein coding transcript. 

Can summarize what I'd call a lot of genic information when exons/CDS have been assigned 
as children. Namely:

* exons
* introns
* utr
* cds
* transcription start and stop
* coding start and stop

This is all setup with the .process_children method (once children are present).

#### GFFLineGene(GFFLine)

Expects (and "manages") GFFLineMrna as class of children

### File oriented

#### GFFReader

turns a gff into a (comment-removed) generator w/ .read_gfffile

nests hierarchical categories and sets up mRNA & gene with their respective 
GFFLineMrna and GFFLineGene classes with .recursive_cluster

## testing

### unit

py.test

Coverage is sad but better than usual. At least for GFFLine and subclasses

### notes

Have been working with ncbi formatted GFF, and non-unit test-as-you-go was 
all done with this. Gff3 only. Planned to add parameterization and GFFConfig
file or so, but haven't done so yet.

## todo

* config for accepting (at least _slightly_)broader range of "gff formats"
* more tests (particularly for GFFReader)
* as marked in place

## usage

Really only ready for use on gff files that contain info on protein-coding genes.

Basic Idea would be: 

```python
import gffreader
filein='my.gff'

# setup reader
gr = gffreader.GFFReader()
# generator from lines in gfffile to GFFLine's
line_gen = gr.read_gfffile(filein)
# organize features under their parents, setup Genes, mRNA
gene_info = gr.recursive_cluster(line_gen)

# 
for gene in gene_info:
    gene.process_children()

```

Likely further accompanied by:

```python
import intervaltree

t = intervaltree.IntervalTree()
# filter to one chromosome per tree, e.g.
on_chr1 = [x for x in gene_info if x.seqid == 'Chr1']
for gene in on_chr1:
    gene.add_to_tree(t)
```
