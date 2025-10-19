# Span_SoftHead_CNN

This is the code for enhanced the [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition](https://aclanthology.org/2023.acl-short.123.pdf) wiht the soft-head of each span.

Using the ConLL 2003 dataset with bert-base-cased pretrained embeddings and a random seed of 16, the best test set F1 score is 92.14


| seed | P. | R. | F1 |
|---------|---------|---------|---------|
| 16 | 92.63 | 91.66 | 92.14 |
| 21 | 92.52 | 91.36 | 91.94 |
| 28 | 92.56 | 91.36 | 91.95 |

Using the ConLL 2003 dataset with roberta-base pretrained embeddings and a random seed of 43, the best test set F1 score is 92.


| seed | P. | R. | F1 |
|---------|---------|---------|---------|
| 42 | 92.61 | 92.92 | 92.76 |
| 27 | 92.40 | 93.02 | 92.71 |
| 18 | 92.69 | 92.26 | 92.48 |
| 33 | 92.20 | 92.26 | 92.23 |
| 37 | 92.56 | 92.69 | 92.62 |
| 22 | 92.00 | 92.63 | 92.32 |
| 29 | 92.05 | 92.65 | 92.35 |
| 20 | 92.48 | 92.53 | 92.50 |
| 49 | 91.86 | 92.53 | 92.19 |
| 13 | 92.69 | 92.26 | 92.48 |
| 43 | 92.85 | 92.14 | 92.49 |
