# PCA

[src/pca.js:24-213](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L24-L213 "Source code on GitHub")

Creates new PCA (Principal Component Analysis) from the dataset

**Parameters**

-   `dataset` **Matrix** dataset or covariance matrix
-   `options` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** 
    -   `options.isCovarianceMatrix` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]**  (optional, default `false`)
    -   `options.center` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]** should the data be centered (subtract the mean) (optional, default `true`)
    -   `options.scale` **\[[boolean](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean)]** should the data be scaled (divide by the standard deviation) (optional, default `false`)

## predict

[src/pca.js:95-106](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L95-L106 "Source code on GitHub")

Project the dataset into the PCA space

**Parameters**

-   `dataset` **Matrix** 

Returns **Matrix** dataset projected in the PCA space

## getExplainedVariance

[src/pca.js:112-118](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L112-L118 "Source code on GitHub")

Returns the proportion of variance for each component

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getCumulativeVariance

[src/pca.js:124-130](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L124-L130 "Source code on GitHub")

Returns the cumulative proportion of variance

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getEigenvectors

[src/pca.js:136-138](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L136-L138 "Source code on GitHub")

Returns the Eigenvectors of the covariance matrix

Returns **Matrix** 

## getEigenvalues

[src/pca.js:144-146](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L144-L146 "Source code on GitHub")

Returns the Eigenvalues (on the diagonal)

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getStandardDeviations

[src/pca.js:152-154](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L152-L154 "Source code on GitHub")

Returns the standard deviations of the principal components

Returns **\[[number](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number)]** 

## getLoadings

[src/pca.js:160-162](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L160-L162 "Source code on GitHub")

Returns the loadings matrix

Returns **Matrix** 

## toJSON

[src/pca.js:168-178](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L168-L178 "Source code on GitHub")

Export the current model to a JSON object

Returns **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** model

## load

[src/pca.js:84-88](https://github.com/mljs/pca/blob/794c550cc253affcdabf42e24f24de0052bf1c05/src/pca.js#L84-L88 "Source code on GitHub")

Load a PCA model from JSON

**Parameters**

-   `model` **[Object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object)** 

Returns **PCA** 
