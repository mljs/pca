# ml-pca

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

Principal component analysis (PCA)

## Installation

`$ npm install ml-pca`

## Usage

```js
const PCA = require('ml-pca');
const dataset = require('ml-dataset-iris').getNumbers();
// dataset is a two-dimensional array where rows represent the samples and columns the features
const pca = new PCA(dataset);
console.log(pca.getExplainedVariance());
/*
[ 0.9246187232017269,
  0.05306648311706785,
  0.017102609807929704,
  0.005212183873275558 ]
*/
const newPoints = [[4.9, 3.2, 1.2, 0.4], [5.4, 3.3, 1.4, 0.9]];
console.log(pca.predict(newPoints)); // project new points into the PCA space
/*
[
  [ -2.830722471866897,
    0.01139060953209596,
    0.0030369648815961603,
    -0.2817812120420965 ],
  [ -2.308002707614927,
    -0.3175048770719249,
    0.059976053412802766,
    -0.688413413360567 ]]
*/
```

## [Documentation](https://mljs.github.io/pca/)

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
