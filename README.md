# scTCAP
single cell Tumor Cell-type Annotation Platform

This workflow enables cell classification of tumor single-cell transcriptome data by inputting single-cell expression matrices. scTCAP classification accuracy is 97%

# Required software and packages

- python = 3.6.8
- pytorch = 1.10.2
- numpy = 1.19.5
- pandas = 1.15
- sklearn = 0.24.1
- scipy = 1.5.4
- tqdm
- argparse

# workflow of scTCAP
![Figure1](https://user-images.githubusercontent.com/23115618/231777952-b291a7d6-ad23-4931-8593-12c8c4a6e0bf.jpg)

# Data prepareration
We used the TPM/CPM value of 10813 genes to predicted cell type of  tumor scRNA-seq data. And scTCAP software does not require pre-training. Users can directly use scTCAP by providing appropriate query data.   
Users need to arrange the gene features of query data according to the gene symbols in scTCAP/data/testdata1000.tsv. Since there are many unexpressed genes in the scRNA-seq data, the expression value of missing genes is set to 0. However, it should be noted that the lack of too many gene will lead to the unsatisfactory classification accuracy of scTCAP.

```shell
testdata e.g.   
       Symbol,BT1300_ACAGCCGAGTGTCCCG,BT1297_CGACTTCCAGTATCTG,CCAATCCGTTCTGGTA.2
       HK1,0,0,0
       PFKM,0,0,0
       PFKP,811.359026369168,0,0
```

# Run scTCAP
After preparing the data, scTCAP can be used to predict the cell type of the query data.
We show an example:
```shell
python scTCAP.py -in ../data/testdata1000.tsv -out ../result/celltype_result.txt -q 0.5

Results e.g.
    CellName, CellType, Score
    BT1300_ACAGCCGAGTGTCCCG, Tumor cell, 0.998
    BT1297_CGACTTCCAGTATCTG, Tumor cell, 0.9888
    CCAATCCGTTCTGGTA.2, Tumor cell, 0.821
    T75_CGAATGTCATGCGCAC.1, Tumor cell, 0.951
    T27_AACTCAGGTACCATCA.1, Tumor cell, 0.927
```
```shell
Arguments
-in <srt>     Input query data name;  
-out <str>    output result name;  
-q <float>    Confidence score cutoff for cell type classification, cells with score below the cutoff will be classified as "unknown" cell-type (default is 0.2).
```
