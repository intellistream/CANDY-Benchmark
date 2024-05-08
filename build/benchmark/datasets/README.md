# The real world datasets

This folder contains all real-world datasets

## Stand-alone MM/AMM datasets

They are used for computing the similarity of two matrixes or the covariance of one matrix
, by matmul(A,B.t()) or matmul(A,A.t()).
| Dataset Name | Application Filed | Size for A,B | If Streaming |
| -------------|--------------------|--------------|------------------|
| AST | Astrophysics | 765*765 | 2 stream |
| BUS | Land Traffic | 4929*10595 | 2 stream |
| DWAVE | Integrated Circuit | 512*512 | 1 stream+1 Static |
| ECO | Economic Models | 207*260 | 1 stream+1 Static |
| QCD(Large)   | Quantum Physics | 49152*49152 | 2 stream |
| QCD(Small)   | Quantum Physics | 3072*3072 | 2 stream |
| RDB | Chemical Engineer | 2048*2048 | 2 stream |
| UTM | Plasma Physics | 1700*1700 | 2 stream |
| ZENOIS | Air Traffic | 2873*2873 | 1 stream+1 Static |

## Note

We have moved the QCD(Large) outside due to size restrictions,
Please use this link:xxxxx

   


