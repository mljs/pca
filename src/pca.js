'use strict';
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');
var SVD = Matrix.DC.SVD;

module.exports = function pca(dataset) {

    var featureNormalize = function(dataset) {
        var means = Stat.matrix.mean(dataset);
        var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
        means = Matrix.rowVector(means);

        var result = dataset.addRowVector(means.neg());
        return result.divRowVector(std);
    };


    var normalizedDataset = featureNormalize(dataset.clone());
    var covarianceMatrix = normalizedDataset.transpose().mmul(normalizedDataset).divS(dataset.rows);

    var target = new SVD(covarianceMatrix, {
        computeLeftSingularVectors: true,
        computeRightSingularVectors: true,
        autoTranspose: false
    });

    return {U: target.leftSingularVectors, S: target.diagonal};
};

