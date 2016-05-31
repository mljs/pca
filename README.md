# ml-pca

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

Principal component analysis (PCA)

## Installation

`$ npm install ml-pca`

## Methods

### new PCA(dataset, options)

__Arguments__

* `dataset` - Data to get the PCA. It must be a two-dimensional array with observations as rows and variables as columns.

__Options__

* `center` - Center the dataset (default: true)
* `scale` - Standardize the dataset, i.e. divide by the standard deviation after centering (default: false)

### predict(dataset)

Project the dataset in the PCA space

__Arguments__

* `dataset` - A Matrix of the dataset to project.

### getExplainedVariance()

Returns the percentage of variance explained by each component.

### getCumulativeVariance()

Returns the cumulative explained variance.

### getStandardDeviations()

Returns the standard deviations of each component.

### getEigenvectors()

Get the eigenvectors of the covariance matrix.

### getEigenvalues()

Get the eigenvalues on the diagonal.

### getLoadings()

Get the loadings matrix (each row is a component and each column is a variable)

## License

  [MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pca.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-pca
[travis-image]: https://img.shields.io/travis/mljs/pca/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/pca
[david-image]: https://img.shields.io/david/mljs/pca.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/pca
[download-image]: https://img.shields.io/npm/dm/ml-pca.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-pca
