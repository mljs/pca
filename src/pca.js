'use strict';
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');
var SVD = Matrix.DC.SVD;

module.exports = PCA;

/**
* Creates new PCA (Principal Component Analysis) from the dataset
* @param {Matrix} dataset
* @constructor
* */
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

/**
* Load a PCA model from JSON
* @oaram {Object} model
* @return {PCA}
* */
PCA.load = function (model) {
    if(model.modelName !== 'PCA')
        throw new RangeError("The current model is invalid!");

    return new PCA(model, true);
};

/**
* Exports the current model to an Object
* @return {Object} model
* */
PCA.prototype.export = function () {
    return {
        modelName: "PCA",
        U: this.U,
        S: this.S
    };
};

/**
* Function that project the dataset into new space of k dimensions,
* this method doesn't modify your dataset.
* @param {Matrix} dataset.
* @param {Number} k - dimensions to project.
* @return {Matrix} dataset projected in k dimensions.
* @throws {RangeError} if k is larger than the number of eigenvector
*                      of the model.
* */
PCA.prototype.project = function (dataset, k) {
    var dimensions = k - 1;
    if(k > this.U.columns)
        throw new RangeError("the number of dimensions must not be larger than " + this.U.columns);

    var X = featureNormalize(Matrix(dataset).clone());
    return X.mmul(this.U.subMatrix(0, this.U.rows - 1, 0, dimensions));
};

/**
* This method returns the percentage variance of each eigenvector.
* @return {vector} percentage variance of each eigenvector.
* */
PCA.prototype.getExplainedVariance = function () {
    var sum = this.S.reduce(function (previous, value) {
        return previous + value;
    });
    return this.S.map(function (value) {
        return value / sum;
    });
};

/**
 * Function that returns the Eigenvectors of the covariance matrix.
 * @returns {Matrix}
 */
PCA.prototype.getEigenvectors = function () {
    return this.U;
};

/**
 * Function that returns the Eigenvalues (on the diagonal).
 * @returns {*}
 */
PCA.prototype.getEigenvalues = function () {
    return this.S;
};

/**
* This method returns a dataset normalized in the following form:
* X = (X - mean) / std
* @param dataset.
* @return A dataset normalized.
* */
function featureNormalize(dataset) {
    var means = Stat.matrix.mean(dataset);
    var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
    means = Matrix.rowVector(means);

    var result = dataset.subRowVector(means);
    return result.divRowVector(std);
}
