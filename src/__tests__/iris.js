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
  [0.261, 0.124, 0.801, 0.524],
];

const expectedLoadingsNIPALS = [
  [0.5211, -0.2693, 0.5804, 0.5649],
  [0.3774, 0.9233, 0.0245, 0.067],
  [0.7196, -0.2444, -0.1421, -0.6343],
  [-0.2613, 0.1235, 0.8014, -0.5236],
];

describe('iris dataset', function () {
  var pca = new PCA(iris, { scale: true, method: 'SVD' });
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
      4,
    );
  });
  it('explained variance', function () {
    expect(pca.getExplainedVariance()).toBeDeepCloseTo(
      [0.7296, 0.2285, 0.03669, 0.00518],
      4,
    );
  });
  it('cumulative variance', function () {
    expect(pca.getCumulativeVariance()).toBeDeepCloseTo(
      [0.7296, 0.9581, 0.9948, 1],
      4,
    );
  });
  it('prediction', function () {
    var pred = pca.predict(iris.slice(0, 2));
    expect(pred.to2DArray()).toBeDeepCloseTo(
      [[-2.257, -0.478, 0.127, -0.024], [-2.074, 0.672, 0.233, -0.103]],
      3,
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
  var pca = new PCA(iris, {
    scale: true,
    method: 'NIPALS',
    nCompNIPALS: 4,
    useCovarianceMatrix: false,
  });

  it('loadings', function () {
    let loadings = pca
      .getLoadings()
      .to2DArray()
      .map((x) => x.map((y) => Math.abs(y)));
    expect(loadings).toBeDeepCloseTo(
      expectedLoadingsNIPALS.map((x) => x.map((y) => Math.abs(y))),
      3,
    );
  });

  it('loadings should be orthogonal', function () {
    let m = pca
      .getLoadings()
      .transpose()
      .mmul(pca.getLoadings())
      .round();
    expect(m.sub(Matrix.eye(4, 4)).sum()).toStrictEqual(0);
  });

  it('eigenvalues', function () {
    let eigenvalues = pca.getEigenvalues();
    expect(eigenvalues.map((x) => Math.sqrt(x))).toBeDeepCloseTo(
      [20.853205, 11.67007, 4.676192, 1.756847],
      6,
    );
  });

  it('scores', function () {
    let scores = pca.predict(iris);
    expect(scores.get(0, 0)).toBeCloseTo(-2.25714118, 6);
    expect(scores.get(0, 1)).toBeCloseTo(0.478423832, 6);
  });

  it('scores may be scaled', function () {
    let scores = pca.predict(iris);
    let eigenvalues = pca.getStandardDeviations();
    let scaledScores = scores.divRowVector(eigenvalues);
    expect(scaledScores.get(0, 0)).toBeCloseTo(-0.1082392451, 6);
  });

  it('X may be recomputed', function () {
    let U = pca.predict(iris);
    let V = pca.getLoadings();
    let S = pca.getEigenvalues();

    // we scale the scores
    let SU = U.divRowVector(S);
    // we recompute X
    let RX = SU.mmul(Matrix.diag(S)).mmul(V);
    expect(RX.get(0, 0)).toBeCloseTo(-0.89767388, 6);
  });

  it('explained variance', function () {
    let R2 = pca.getExplainedVariance();
    expect(R2).toBeDeepCloseTo(
      [0.729624454, 0.228507618, 0.036689219, 0.005178709],
      4,
    );
  });
});

describe('iris dataset and nipals default nCompNIPALS', function () {
  var pca = new PCA(iris, {
    scale: true,
    method: 'NIPALS',
    useCovarianceMatrix: false,
  });

  it('eigenvalues', function () {
    let sd = pca.getStandardDeviations();
    expect(sd).toBeDeepCloseTo([20.853205, 11.67007], 6);
  });

  it('prediction', () => {
    var pred = pca.predict(iris.slice(0, 2));
    expect(pred.to2DArray()).toBeDeepCloseTo(
      [[-2.257, 0.478], [-2.074, -0.672]],
      3,
    );
  });
});
