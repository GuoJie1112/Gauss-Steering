# Machine Learning Training on Gaussian steering

To overcome the computational limitations of traditional Gaussian steering quantification methods, we utilize  machine learning serving as a powerful tool for detecting the steering of Gaussian states.

# Training sample of Gaussian steering detection in continuous variable system

To train the Gaussian steering classifiers for an $(n+m)$-mode CV system, it is necessary to collect covariance matrices $\Gamma_\rho $ of $(n+m)$-mode Gaussian states $\rho$ as training samples and to extract relevant features from these matrices. We can generate covariance matrix $\Gamma_\rho $ randomly in a certain way.

The training samples are generated as follows. We create $ l $ random covariance matrices by the way and derive labels based on the property of  $ \mathcal{J} $ : for any $ (n+m) $-mode Gaussian state $ \rho $, $ \mathcal{J}(\Gamma_{\rho}) = 0 $ if and only if $ \rho $ is unsteerable from $ A $ to $ B $ through Alice's Gaussian measurements; conversely, $ \mathcal{J}(\Gamma_{\rho}) > 0 $ indicates steering. This allows us to label the samples accordingly. If the covariance matrix $ \Gamma_{\rho} $ corresponds to a steerable Gaussian state $ \rho $, the label $ y = 1 $; otherwise, $ y = 0 $. Consequently, we collect a batch of labeled data $\mathcal{D}=  \{(\Gamma_{i}, y_{i})\}_{i=1}^{l} $, consisting of covariance matrices for both steerable and unsteerable Gaussian states. This labeled dataset serves as the foundation for training the classifiers.

Notably, during the process of constructing the dataset $\mathcal{D}=  \{(\Gamma_{i}, y_{i})\}_{i=1}^{l} $, we observe that the number of unsteerable Gaussian states far exceeded that of steerable states. Achieving a balance between the numbers of steerable and unsteerable states required extremely high computational costs. This made collecting samples labeled as +1 (indicating steerable states) especially challenging, particularly as the number of samples increased. Additionally, it is important to note that the covariance matrices corresponding to Gaussian states of different modes are completely random and independent.

To better reflect real conditions and to develop models that are more suited to the practical imbalance in the distribution of steerable and unsteerable Gaussian states, we divided the dataset  $\mathcal{D}= \{(\Gamma_{i}, y_{i})\}_{i=1}^{l} $ construction process into the following approaches:
1.Balanced dataset.
2.Naturally imbalanced datasets.
3.Imbalanced datasets via augmentation strategy.

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 0.4.0 
