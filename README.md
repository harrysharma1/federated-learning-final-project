# Federated Learning Security Analysis

This repository will host mostly if not all the code used for this project either directly on here (will be underscored e.g. [dlg_notebook](https://github.com/harrysharma1/federated-learning-security/tree/main/dlg_notebooks)), or as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) (will be dashed e.g. [federated-learning-analysis](https://github.com/harrysharma1/federated-learning-analysis/tree/884b9a62deac353a49ea6e9e530d853e27873e7f).

## Table of Contents

### [Deep Leakage Example (from MIT's Han Lab).](https://github.com/mit-han-lab/dlg)

### [The Example I created as a test.](https://github.com/harrysharma1/federated-learning-analysis)

#### Features
- [x] Re-implementation of [DLG](https://github.com/mit-han-lab/dlg) to understand how it works.
- [x] Add Check for grayscale (to then convert to RGB) that was not in the original implementation.
- [X] Showcase on README.

#### TODO
- [x] DONE

### [Web application to highlight this research.](https://github.com/harrysharma1/federated-learning-results)

#### Features
- [x] Using [my tweaked DLG](https://github.com/harrysharma1/federated-learning-analysis) to showcase the effect of poor security on CIFAR Dataset.
- [x] Add a multiple range version, that will also showcase this.
- [x] Add randomise button for both single and multiple versions.
- [x] Showcase similarity with measures of Structural Similarity Index Measure (SSIM), Mean Squar Error (MSE), and Peak Signal-To-Noise Ratio.

#### TODO
- [ ] Add upload for personal image to test this out.
- [ ] Add text based version to showcase this.
- [ ] more to come...

### [Sentiment analysis using general ML method, may remove in the future but gave insipiration to add text based training showcase for DLG.](https://github.com/harrysharma1/sentiment-analysis)

## Acknowledgements

- [DLG](https://github.com/mit-han-lab/dlg)
