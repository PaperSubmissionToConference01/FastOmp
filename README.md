# Fast and Fair Feature Selection

This repository provides a python implementation of our submission titled ''Fast and Fair Feature Selection''. This is a temporary anonymized repository.

The ProPublica COMPAS dataset, and the ISK implementation for this dataset were taken from this repository: https://github.com/nina-gh/procedurally_fair_learning/tree/master/fair_feature_selection

The ProPublica COMPAS dataset can be originally found here: https://github.com/propublica/compas-analysis

### Dependencies 

[numpy, scipy](https://www.scipy.org/scipylib/download.html), [pandas](https://pandas.pydata.org/) and [sklearn](http://scikit-learn.org/)


### Demo

By running demo.py, you can see a demo of our experiments, applied on a real-world dataset.

The demo showcases the perofmance of the tested algorithms on the real-world biomedical dataset [4]. The same code can be used to showcase performance on the synthetic dataset.

Please note that some of the parameters of the algorithms are not tuned to best performance.

The demo saves the results to text files for the tested algorithms. 


### Files

1. code/demo.py
2. code/algos.py
3. code/model.py	

4. data/healthdata.csv	
5. data/synthetic.csv				


### File Descriptions

#### Code

- File 1: File 1 contains a demo of our experiments for Fait Feature Selection.

- File 2: File 2 contains an implementation of the tested algorithms.

- File 3: File 3 contains the code for the oracl functions, including the code to generate the GLMs.


#### Data

- Files 4 and 5: File 4 constains the bio-medical data for the experiments. File 5 contains the synthetic dataset for the experiments.


### References

[1] Andreas Krause and Volkan Cevher. Submodular dictionary selection for sparse representation. ICML (2010)

[2] Nina Grgić-Hlača, Muhammad Bilal Zafar, Krishna P. Gummadi, and Adrian Weller. Beyond Distributive Fairness in Algorithmic Decision Making: Feature Selection for Procedurally Fair Learning AAAI (2018).

[3] Rishabh Iyer and Jeff A. Bilmes. Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints. NIPS (2013).

[4] Cancer Genome Atlas Research Network, John N Weinstein, Eric A Collisson, Gordon B379Mills, Kenna R Mills Shaw, Brad A Ozenberger, Kyle Ellrott, Ilya Shmulevich, Chris Sander, and Joshua M Stuart. The cancer genome atlas pan-cancer analysis project. Nature Genetics, 45(10):1113–20, 2013.


