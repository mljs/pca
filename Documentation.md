# PCA

[src/pca.js:19-216](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L19-L216 "Source code on GitHub")

Class representing a PCA

## constructor

[src/pca.js:28-80](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L28-L80 "Source code on GitHub")

Creates new PCA (Principal Component Analysis) from the dataset

**Parameters**

-   `dataset` **Matrix** dataset or covariance matrix
-   `options` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** 
    -   `options.isCovarianceMatrix` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]**  (optional, default `false`)
    -   `options.center` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]** should the data be centered (subtract the mean) (optional, default `true`)
    -   `options.scale` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]** should the data be scaled (divide by the standard deviation) (optional, default `false`)

## predict

[src/pca.js:98-109](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L98-L109 "Source code on GitHub")

Project the dataset into the PCA space

**Parameters**

-   `dataset` **Matrix** 

Returns **Matrix** dataset projected in the PCA space

## getExplainedVariance

[src/pca.js:115-121](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L115-L121 "Source code on GitHub")

Returns the proportion of variance for each component

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getCumulativeVariance

[src/pca.js:127-133](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L127-L133 "Source code on GitHub")

Returns the cumulative proportion of variance

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getEigenvectors

[src/pca.js:139-141](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L139-L141 "Source code on GitHub")

Returns the Eigenvectors of the covariance matrix

Returns **Matrix** 

## getEigenvalues

[src/pca.js:147-149](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L147-L149 "Source code on GitHub")

Returns the Eigenvalues (on the diagonal)

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getStandardDeviations

[src/pca.js:155-157](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L155-L157 "Source code on GitHub")

Returns the standard deviations of the principal components

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getLoadings

[src/pca.js:163-165](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L163-L165 "Source code on GitHub")

Returns the loadings matrix

Returns **Matrix** 

## toJSON

[src/pca.js:171-181](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L171-L181 "Source code on GitHub")

Export the current model to a JSON object

Returns **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** model

## load

[src/pca.js:87-91](https://github.com/mljs/pca/blob/4bef6ec5940ca5bcc84b16f4015a68dc7821035a/src/pca.js#L87-L91 "Source code on GitHub")

Load a PCA model from JSON

**Parameters**

-   `model`  

Returns **PCA** 
