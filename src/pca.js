'use strict';
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');
var SVD = Matrix.DC.SVD;

module.exports = PCA;

function PCA(dataset, reload) {

    if (reload) {
        this.U = dataset.U;
        this.S = dataset.S;
    } else {
        if (!Matrix.isMatrix(dataset)) {
            dataset = new Matrix(dataset);
        } else {
            dataset = dataset.clone();
        }
        var normalizedDataset = featureNormalize(dataset);
        var covarianceMatrix = normalizedDataset.transpose().mmul(normalizedDataset).divS(dataset.rows);

        var target = new SVD(covarianceMatrix, {
            computeLeftSingularVectors: true,
            computeRightSingularVectors: true,
            autoTranspose: false
        });

        this.U = target.leftSingularVectors;
        this.S = target.diagonal;
    }
}

PCA.load = function (model) {
    if(model.modelName !== 'PCA')
        throw new RangeError("The current model is invalid!");

    return new PCA(model, true);
};

PCA.prototype.export = function () {
    var model = {
        modelName: "PCA",
        U: this.U,
        S: this.S
    };

    return model;
};

PCA.prototype.project = function (dataset, dimensions) {
    var dim = dimensions - 1;
    if(dimensions > this.U.columns)
        throw new RangeError("the number of dimensions must not be larger than " + this.U.columns);

    var X = featureNormalize(Matrix(dataset).clone());
    return X.mmul(this.U.subMatrix(0, this.U.rows - 1, 0, dim));
};

PCA.prototype.getExplainedVariance = function () {
    var sum = this.S.reduce(function (previous, value) {
        return previous + value;
    });
    return this.S.map(function (value) {
        return value / sum;
    });
};

function featureNormalize(dataset) {
    var means = Stat.matrix.mean(dataset);
    var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
    means = Matrix.rowVector(means);

    var result = dataset.subRowVector(means);
    return result.divRowVector(std);
}
