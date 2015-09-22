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

* `dataset` - Data to get the PCA.

__Options__

* `standardize` - Standardize the dataset, i.e. divide by the standard deviation after centering (default: false)

__Example__

```js
var dataset = [ ... ];

var pca = new PCA(dataset);
```

### project(dataset, k)

Project the dataset over k dimensions 

__Arguments__

* `dataset` - A Matrix of the dataset.
* `k` - Number of dimensions to be projected.

__Example__

```js
var data = [ ... ];

var projectedData = pca.project(data, k);
```

### getExplainedVariance()

Returns the percentage of variance of each vector of the PCA.

### getEigenvectors()

Get the eigenvectors of the covariance matrix.

### getEigenvalues()

Get the eigenvalues on the diagonal.

### export()

Exports the actual PCA to an Javascript Object.

### load(model)

Returns a new PCA with the given model.

__Arguments__

* `model` - Javascript Object generated from export() function.

## Authors

- [Jefferson Hernandez](https://github.com/JeffersonH44)

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
