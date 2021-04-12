# ml-pca

Principal component analysis (PCA).

<h3 align="center">

  <a href="https://www.zakodium.com">
    <img src="https://www.zakodium.com/brand/zakodium-logo-white.svg" width="50" alt="Zakodium logo" />
  </a>

  <p>
    Maintained by <a href="https://www.zakodium.com">Zakodium</a>
  </p>

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![npm download][download-image]][download-url]

</h3>

## Installation

`$ npm install ml-pca`

## Usage

```js
const { PCA } = require('ml-pca');
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
const newPoints = [
  [4.9, 3.2, 1.2, 0.4],
  [5.4, 3.3, 1.4, 0.9],
];
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

## [API Documentation](https://mljs.github.io/pca/)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pca.svg
[npm-url]: https://npmjs.org/package/ml-pca
[ci-image]: https://github.com/mljs/pca/workflows/Node.js%20CI/badge.svg?branch=master
[ci-url]: https://github.com/mljs/pca/actions?query=workflow%3A%22Node.js+CI%22
[download-image]: https://img.shields.io/npm/dm/ml-pca.svg
[download-url]: https://npmjs.org/package/ml-pca
