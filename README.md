# scTCAP
single cell Tumor Cell-type Annotation Platform

This workflow enables cell classification of tumor single-cell transcriptome data by inputting single-cell expression matrices. scTCAP classification accuracy is 97%

# Required software and packages

- python = 3.6.8
- pytorch = 1.10.2
- numpy = 1.19.5
- argparse

# workflow of scTCAP
![739509798323892224](https://user-images.githubusercontent.com/23115618/178656464-52ff76ca-fc6e-44ff-a38d-ab2609374426.jpg)

# Data prepareration
scTCAP软件无需进行模型预训练，采用10813个基因特征条目的TPM/CPM值进行细胞类型分类。因此需要使用者对分类数据的基因特征按照scTCAP/data/testdata1000.tsv中的基因特征symbol进行排列。对于缺失的基因条目，默认为该队列中未检测到相关特征的表达，因此赋值为0。值得注意的是由于基因特征中存在LncRNA等非编码序列信息，因此我们在测试集合中采用基因symbol形式进行展示特征基因。
```shell
testdata e.g.   
       Symbol,BT1300_ACAGCCGAGTGTCCCG,BT1297_CGACTTCCAGTATCTG,CCAATCCGTTCTGGTA.2
       HK1,0,0,0
       PFKM,0,0,0
       PFKP,811.359026369168,0,0
```

# Run scTCAP
完成数据的准备后即可使用scTCAP模型对数据的细胞类型进行分类，我们给出了1000个细胞的测试数据：scTCAP/data/testdata1000.tsv。
```shell
python scTCAP.py -in ../data/testdata1000.tsv -out ../result/celltype_result.txt -q 0.3 -sep \t
  
result e.g.
    CellName, CellType
    BT1300_ACAGCCGAGTGTCCCG, Tumor cell
    BT1297_CGACTTCCAGTATCTG, Tumor cell
    CCAATCCGTTCTGGTA.2, Tumor cell
    T75_CGAATGTCATGCGCAC.1, Tumor cell
    T27_AACTCAGGTACCATCA.1, Tumor cell
```
```shell
Arguments
-in <srt>     
-out <str>
-q <float>    细胞类型分类的可靠性打分.(default 0.2)
-sep <str>    输入文件的分隔符号.(default "\t")
```
