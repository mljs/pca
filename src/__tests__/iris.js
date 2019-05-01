// Ref: http://www.r-bloggers.com/computing-and-visualizing-pca-in-r/

import { Matrix } from 'ml-matrix';
import { getNumbers } from 'ml-dataset-iris';
import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';

import { PCA } from '../pca';

expect.extend({ toBeDeepCloseTo });

const iris = getNumbers();

const expectedLoadings = [
  [0.521, 0.269, 0.58, 0.565],
  [0.377, 0.923, 0.024, 0.067],
  [0.72, 0.244, 0.142, 0.634],
  [0.261, 0.124, 0.801, 0.524]
];

describe('iris dataset', function () {
  var pca = new PCA(iris, { scale: true, useCovarianceMatrix: false });
  it('loadings', function () {
    var loadings = pca
      .getLoadings()
      .to2DArray()
      .map((x) => x.map((y) => Math.abs(y)));
    expect(loadings).toBeDeepCloseTo(expectedLoadings, 3);
  });
  it('standard deviation', function () {
    expect(pca.getStandardDeviations()).toBeDeepCloseTo(
      [1.7084, 0.956, 0.3831, 0.1439],
      4
    );
  });
  it('explained variance', function () {
    expect(pca.getExplainedVariance()).toBeDeepCloseTo(
      [0.7296, 0.2285, 0.03669, 0.00518],
      4
    );
  });
  it('cumulative variance', function () {
    expect(pca.getCumulativeVariance()).toBeDeepCloseTo(
      [0.7296, 0.9581, 0.9948, 1],
      4
    );
  });
  it('prediction', function () {
    var pred = pca.predict(iris.slice(0, 2));
    expect(pred.to2DArray()).toBeDeepCloseTo(
      [[-2.257, -0.478, 0.127, -0.024], [-2.074, 0.672, 0.233, -0.103]],
      3
    );
  });
  it('inverting scaled', () => {
    var input = iris.slice(0, 2);
    var pred = pca.predict(input);

    var inv = pca.invert(pred);

    expect(inv.to2DArray()).toBeDeepCloseTo(input);
  });
  it('inverting not scaled', () => {
    var dataset = [[1, 2, 3], [0, 3, 5], [2, 2, 2]];
    var newpca = new PCA(dataset);
    var pred = newpca.predict(dataset);

    var inv = newpca.invert(pred);

    expect(inv.to2DArray()).toBeDeepCloseTo(dataset);
  });
});

describe('iris dataset with provided covariance matrix', function () {
  var dataset = new Matrix(iris);
  var mean = dataset.mean('column');
  var stdevs = dataset.standardDeviation('column', { mean });
  dataset.subRowVector(mean).divRowVector(stdevs);
  var covarianceMatrix = dataset
    .transpose()
    .mmul(dataset)
    .divS(dataset.rows - 1);
  var pca = new PCA(covarianceMatrix, { isCovarianceMatrix: true });
  it('loadings', function () {
    var loadings = pca
      .getLoadings()
      .to2DArray()
      .map((x) => x.map((y) => Math.abs(y)));
    expect(loadings).toBeDeepCloseTo(expectedLoadings, 3);
  });
});

describe('iris dataset with computed covariance matrix', function () {
  var pca = new PCA(iris, { scale: true, useCovarianceMatrix: true });
  it('loadings', function () {
    var loadings = pca
      .getLoadings()
      .to2DArray()
      .map((x) => x.map((y) => Math.abs(y)));
    expect(loadings).toBeDeepCloseTo(expectedLoadings, 3);
  });
});
