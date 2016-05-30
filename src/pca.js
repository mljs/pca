'use strict';

const Matrix = require('ml-matrix');
const SVD = Matrix.DC.SVD;
const Stat = require('ml-stat');

const defaultOptions = {
    standardize: false
};

class PCA {
    /**
     * Creates new PCA (Principal Component Analysis) from the dataset
     * @param {Matrix} dataset
     * @param {Object} options - options for the PCA algorithm
     * @param {boolean} reload - for load purposes
     * @param {Object} model - for load purposes
     * @constructor
     * */
    constructor(dataset, options, reload, model) {
        if (reload) {
            this.U = model.U;
            this.S = model.S;
            this.means = model.means;
            this.std = model.std;
            this.standardize = model.standardize
        } else {
            options = Object.assign({}, defaultOptions, options);

            this.standardize = !!options.standardize;

            if (!Matrix.isMatrix(dataset)) {
                dataset = new Matrix(dataset);
            } else {
                dataset = dataset.clone();
            }

            var normalization = adjust(dataset, this.standardize);
            var normalizedDataset = normalization.result;

            var covarianceMatrix = normalizedDataset.transpose().mmul(normalizedDataset).divS(dataset.rows);

            var target = new SVD(covarianceMatrix, {
                computeLeftSingularVectors: true,
                computeRightSingularVectors: true,
                autoTranspose: false
            });

            this.U = target.leftSingularVectors;
            this.S = target.diagonal;
            this.means = normalization.means;
            this.std = normalization.std;
        }
    }

    /**
     * Load a PCA model from JSON
     * @oaram {Object} model
     * @return {PCA}
     */
    static load(model) {
        if (model.name !== 'PCA')
            throw new RangeError('Invalid model: ' + model.name);
        return new PCA(null, null, true, model);
    }

    /**
     * Exports the current model to an Object
     * @return {Object} model
     */
    toJSON() {
        return {
            name: 'PCA',
            U: this.U,
            S: this.S,
            means: this.means,
            std: this.std,
            standardize: this.standardize
        };
    }

    /**
     * Projects the dataset into new space of k dimensions,
     * this method doesn't modify your dataset.
     * @param {Matrix} dataset.
     * @param {Number} k - dimensions to project.
     * @return {Matrix} dataset projected in k dimensions.
     * @throws {RangeError} if k is larger than the number of eigenvector
     *                      of the model.
     */
    project(dataset, k) {
        var dimensions = k - 1;
        if (k > this.U.columns)
            throw new RangeError("the number of dimensions must not be larger than " + this.U.columns);

        if (!Matrix.isMatrix(dataset)) {
            dataset = new Matrix(dataset);
        } else {
            dataset = dataset.clone();
        }

        var X = adjust(dataset, this.standardize, this.means, this.std).result;
        return X.mmul(this.U.subMatrix(0, this.U.rows - 1, 0, dimensions));
    }

    /**
     * Returns the percentage variance of each eigenvector.
     * @return {Number}
     */
    getExplainedVariance() {
        var sum = this.S.reduce((previous, value) => previous + value);
        return this.S.map(value => value / sum);
    }

    /**
     * Returns the Eigenvectors of the covariance matrix.
     * @returns {Matrix}
     */
    getEigenvectors() {
        return this.U;
    }

    /**
     * Returns the Eigenvalues (on the diagonal).
     * @returns {*}
     */
    getEigenvalues() {
        return this.S;
    }
}

function adjust(dataset, standarize, means, std) {
    if (!means) means = Stat.matrix.mean(dataset);
    if (!std) std = standarize ? Stat.matrix.standardDeviation(dataset, means, true) : undefined;

    var result = dataset.subRowVector(means);
    return {
        result: standarize ? result.divRowVector(std) : result,
        means: means,
        std: std
    }
}

module.exports = PCA;
