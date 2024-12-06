# Experiment 3: Reproduction of LMM-CMA-ES

## 1. Introduction
LMM-CMA-ES, which stands for local meta-model covariance matrix adaptation evolution strategy, is a metamodel enhancement proposed by Nikolaus Hansen for his CMA-ES algorithm. One of the most important contributions of this publication is the introduction of Approximate Ranking Metamodel. This metamodel has also been adopted by Konrad Krawczyk during his work on JADE, where he used the same metamodel management algorithm but replaced LWR with KNN.

Reimplementation of this metamodel on CMA-ES has been proposed as the first step in the research process of SOFES project. As in the case of Konrad Krawczyk's JADE with ARM, I'm using KNN metamodel.

Now to check if the implementation is done correctly, the results from lmm-cma-es shall be reproduced.

The experiment will be considered successful if the results of reproduction (number of objective function evaluation and the ratio of such evaluations with and without the metamodel) are close to those reported in the publication.

## 2. Problem Definition and Algorithm

### 2.1 Task Definition

Precisely define the problem you are addressing (i.e. formally specify the inputs and outputs). Elaborate on why this is an interesting and important problem.

### 2.2 Algorithm Definition

Describe in reasonable detail the algorithm you are using to address this problem. A psuedocode description of the algorithm you are using is frequently useful. Trace through a concrete example, showing how your algorithm processes this example. The example should be complex enough to illustrate all of the important aspects of the problem but simple enough to be easily understood. If possible, an intuitively meaningful example is better than one with meaningless symbols.

## 3. Experimental Evaluation

### 3.1 Methodology

What are criteria you are using to evaluate your method? What specific hypotheses does your experiment test? Describe the experimental methodology that you used. What are the dependent and independent variables? What is the training/test data that was used, and why is it realistic or interesting? Exactly what performance data did you collect and how are you presenting and analyzing it? Comparisons to competing methods that address the same problem are particularly useful.

### 3.2 Results

Present the quantitative results of your experiments. Graphical data presentation such as graphs and histograms are frequently better than tables. What are the basic differences revealed in the data. Are they statistically significant?

### 3.3 Discussion

Is your hypothesis supported? What conclusions do the results support about the strengths and weaknesses of your method compared to other methods? How can the results be explained in terms of the underlying properties of the algorithm and/or the data.

## 4. Related Work

Answer the following questions for each piece of related work that addresses the same or a similar problem. What is their problem and method? How is your problem and method different? Why is your problem and method better?

## 5. Future Work

What are the major shortcomings of your current method? For each shortcoming, propose additions or enhancements that would help overcome it.

## 6. Conclusion
Briefly summarize the important results and conclusions presented in the paper. What are the most important points illustrated by your work? How will your results improve future research and applications in the area?

## Bilbiography
- [1] lmm cma-es
- [2] krawczyk
Be sure to include a standard, well-formated, comprehensive bibliography with citations from the text referring to previously published papers in the scientific literature that you utilized or are related to your work.