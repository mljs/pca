'use strict';

// Ref: http://www.r-bloggers.com/computing-and-visualizing-pca-in-r/

const Matrix = require('ml-matrix').Matrix;
const Stat = require('ml-stat/matrix');
const mean = Stat.mean;
const stdev = Stat.standardDeviation;

const iris = require('ml-dataset-iris').getNumbers();
const PCA = require('..');

const expectedLoadings = [
    [0.521, 0.269, 0.580, 0.565],
    [0.377, 0.923, 0.024, 0.067],
    [0.720, 0.244, 0.142, 0.634],
    [0.261, 0.124, 0.801, 0.524]
];

describe('iris dataset', function () {
    var pca = new PCA(iris, {scale: true, useCovarianceMatrix: false});
    it('loadings', function () {
        checkLoadings(pca);
    });
    it('standard deviation', function () {
        pca.getStandardDeviations().should.approximatelyDeep([1.7084, 0.9560, 0.3831, 0.1439], 1e-4);
    });
    it('explained variance', function () {
        pca.getExplainedVariance().should.approximatelyDeep([0.7296, 0.2285, 0.03669, 0.00518], 1e-4);
    });
    it('cumulative variance', function () {
        pca.getCumulativeVariance().should.approximatelyDeep([0.7296, 0.9581, 0.9948, 1], 1e-4);
    });
    it('prediction', function () {
        var pred = pca.predict(iris.slice(0,2));
        pred.should.approximatelyDeep([
            [-2.257, -0.478, 0.127, -0.024],
            [-2.074, 0.672, 0.233, -0.103]
        ], 1e-3);
    });
});

describe('iris dataset with provided covariance matrix', function () {
    var dataset = new Matrix(iris);
    var means = mean(dataset);
    var stdevs = stdev(dataset, means, true);
    dataset.subRowVector(means).divRowVector(stdevs);
    var covarianceMatrix = dataset.transpose().mmul(dataset).divS(dataset.rows - 1);
    var pca = new PCA(covarianceMatrix, {isCovarianceMatrix: true});
    it('loadings', function () {
        checkLoadings(pca);
    });
});

describe('iris dataset with computed covariance matrix', function () {
    var pca = new PCA(iris, {scale: true, useCovarianceMatrix: true});
    let predict = pca.predict(iris);
    it('loadings', () => {
        checkLoadings(pca);
    });
});

describe('iris dataset with nipals algorithm', () => {
    var pca = new PCA(iris, {scale: true, center: true, algorithm: 'nipals', maxIterations: 100, nComponents: 4});
    it('loadins', () => {
        var loadings = pca.getLoadings().map(x => x .map(y => Math.abs(y)));
        loadings.should.approximatelyDeep([
            [0.5210662, 0.37741676,  0.7195674, 0.2612842],
            [0.2693468, 0.92329601, 0.2443815,  0.1235089],
            [0.5804131, 0.02449134, 0.1421287,  0.8014488],
            [0.5648566, 0.06694208, 0.6342712, 0.5235990]
        ], 1e-3);
    });
});
function checkLoadings(pca) {
    var loadings = pca.getLoadings().map(x => x .map(y => Math.abs(y)));
    loadings.should.approximatelyDeep(expectedLoadings, 1e-3);
}
