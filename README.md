# Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook

## Early access

Preliminary version available at [https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html).

## Motivations

Machine learning for credit card fraud detection (ML for CCFD) has become an active research field. This is illustrated by the [remarkable amount of publications on the topic in the last decade](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/MachineLearningForFraudDetection.html). 

It makes no doubt that the integration of machine learning techniques in payment card fraud detection systems has greatly improved their ability to more efficiently detect frauds. At the same time, a major issue in this new research field is the lack of reproducibility. There do not exist any recognized benchmarks, nor methodologies, to compare and assess the proposed techniques.

This book aims at making a first step in this direction. All the techniques and results provided in this book are reproducible. Sections that include code are Jupyter notebooks, which can be executed either locally, or on the cloud using [Google Colab](https://colab.research.google.com/) or [Binder](https://mybinder.org/). 

The intended audience is students or professionals, interested in the specific problem of credit card fraud detection from a practical point of view. More generally, we think the book is also of interest for data practitioners and data scientists dealing with machine learning problems that involve sequential data and/or imbalanced classification problems.

Provisional table of content: 

* Chapter 1: Book overview
* Chapter 2: Background
* Chapter 3: Getting started
* Chapter 4: Performance metrics
* Chapter 5: Model selection
* Chapter 6: Imbalanced learning
* Chapter 7: Deep learning
* Chapter 8: Interpretability*

(*): Not yet published. 

## Current draft

The writing of the book is ongoing. We provide through this Github repository an early access to the book. As of January 2022, the first seven chapters are made available. 

The online version of the current draft of this book is available [here](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).

Any comment or suggestion is welcome. We recommend using Github issues to start a discussion on a topic, and to use pull requests for fixing typos. 


## Compiling the book

In order to read and/or execute this book on your computer, you will need to clone this repository and compile the book. 

This book is a Jupyter book. You will therefore first need to [install Jupyter Book](https://jupyterbook.org/intro.html#install-jupyter-book).

The compilation was tested with the following package versions:

```
sphinxcontrib-bibtex==2.2.1
Sphinx==4.2.0
jupyter-book==0.11.2
```

Once done, this is a two-step process:

1. Clone this repository:

```
git clone https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
```

2. Compile the book

```
jupyter-book build fraud-detection-handbook
```

The book will be available locally at `fraud-detection-handbook/_build/html/index.html`.

## License

The code in the notebooks is released under a [GNU GPL v3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html). The prose and pictures are released under a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).


If you wish to cite this book, you may use the following:

<pre>
@book{leborgne2022fraud,
title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
author={Le Borgne, Yann-A{\"e}l and Siblini, Wissam and Lebichot, Bertrand and Bontempi, Gianluca},
url={https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook},
year={2022},
publisher={Universit{\'e} Libre de Bruxelles}
}
</pre>

## Authors

* [Yann-Aël Le Borgne](https://yannael.github.io/) (Contact author - yann-ael.le.borgne@ulb.be) - [Machine Learning Group - Université Libre de Bruxelles, Belgium](http://mlg.ulb.ac.be). 
* [Wissam Siblini](https://www.linkedin.com/in/wissam-siblini) - [Machine Learning Research - Worldline Labs](https://worldline.com)
* [Bertrand Lebichot](https://b-lebichot.github.io/) - [Interdisciplinary Centre for Security, Reliability and Trust  - Université du Luxembourg, Luxembourg](https://wwwfr.uni.lu/snt)
* [Gianluca Bontempi](https://mlg.ulb.ac.be/wordpress/members-2/gianluca-bontempi/) - [Machine Learning Group - Université Libre de Bruxelles, Belgium](http://mlg.ulb.ac.be)


## Acknowledgments

This book is the result of ten years of collaboration between the [Machine Learning Group, Université Libre de Bruxelles, Belgium](http://mlg.ulb.ac.be) and [Worldline](https://worldline.com). 

* ULB-MLG, Principal investigator: Gianluca Bontempi
* Worldline, R&D Manager: Frédéric Oblé

We wish to thank all the colleagues who worked on this topic during this collaboration: Olivier Caelen (ULB-MLG/Worldline), Fabrizio Carcillo (ULB-MLG), Guillaume Coter (Worldline), Andrea Dal Pozzolo (ULB-MLG), Jacopo De Stefani (ULB-MLG), Rémy Fabry (Worldline), Liyun He-Guelton (Worldline), Gian Marco Paldino (ULB-MLG), Théo Verhelst (ULB-MLG).

The collaboration was made possible thanks to [Innoviris](https://innoviris.brussels), the Brussels Region Institute for Research and Innovation, through a series of grants which started in 2012 and ended in 2021.

* 2018 to 2021. *DefeatFraud: Assessment and validation of deep feature engineering and learning solutions for fraud detection*. Innoviris Team Up Programme. 
* 2015 to 2018. *BruFence: Scalable machine learning for automating defense system*. Innoviris Bridge Programme.
* 2012 to 2015. *Adaptive real-time machine learning for credit card fraud detection*. Innoviris Doctiris Programme. 

The collaboration is continuing in the context of the [Data Engineering for Data Science (DEDS) project](https://deds.ulb.ac.be/) - under the Horizon 2020 - Marie Skłodowska-Curie Innovative Training Networks (H2020-MSCA-ITN-2020) framework.