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

const expectedLoadingsNIPALS = [
  [0.5211, -0.2693, 0.5804, 0.5649],
  [0.3774, 0.9233, 0.0245, 0.067],
  [0.7196, -0.2444, -0.1421, -0.6343],
  [-0.2613, 0.1235, 0.8014, -0.5236]
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

describe('iris dataset and nipals', function () {
  var pca = new PCA(iris, { scale: true,
    useNIPALS: true,
    nCompNIPALS: 4,
    useCovarianceMatrix: false });

  it('loadings', function () {
    let loadings = pca
      .getLoadings()
      .to2DArray()
      .map((x) => x.map((y) => Math.abs(y)));
    expect(loadings).toBeDeepCloseTo(
      expectedLoadingsNIPALS.map((x) =>
        x.map((y) => Math.abs(y))), 3);
  });
  it('eigenvalues', function () {
    let eigenvalues = pca
      .getEigenvalues()
      .to1DArray();
    expect(eigenvalues).toBeDeepCloseTo([20.853205, 11.670070, 4.676192, 1.756847], 6);
  });
});

describe('iris dataset and nipals default', function () {
  var pca = new PCA(iris, { scale: true,
    useNIPALS: true,
    useCovarianceMatrix: false });

  it('eigenvalues', function () {
    let eigenvalues = pca
      .getEigenvalues()
      .to1DArray();
    expect(eigenvalues).toBeDeepCloseTo([20.853205, 11.670070], 6);
  });
});

describe('iris dataset and nipals default without scaling scores', function () {
  var pca = new PCA(iris, { scale: true,
    useNIPALS: true,
    scaleScores: true,
    useCovarianceMatrix: false });

  it('eigenvalues', function () {
    let eigenvalues = pca
      .getEigenvalues()
      .to1DArray();
    expect(eigenvalues).toBeDeepCloseTo([20.853205, 11.670070], 6);
  });

  it('scores', function () {
    let scores = pca.predict(iris);
    expect(scores.get(0, 0)).toBeCloseTo(-2.25714118, 6);
    expect(scores.get(0, 1)).toBeCloseTo(0.478423832, 6);
  });
});

