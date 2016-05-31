'use strict';

const Matrix = require('ml-matrix');
const SVD = Matrix.DC.SVD;
const Stat = require('ml-stat').matrix;
const mean = Stat.mean;
const stdev = Stat.standardDeviation;

const defaultOptions = {
    center: true,
    scale: false
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
            this.center = model.center;
            this.scale = model.scale;
            this.means = model.means;
            this.stdevs = model.stdevs;
            this.U = Matrix.checkMatrix(model.U);
            this.S = model.S;
        } else {
            options = Object.assign({}, defaultOptions, options);

            this.center = !!options.center;
            this.scale = !!options.scale;

            dataset = new Matrix(dataset);

            if (this.center) {
                const means = mean(dataset);
                const stdevs = this.scale ? stdev(dataset, means, true) : null;
                this.means = means;
                dataset.subRowVector(means);
                if (this.scale) {
                    this.stdevs = stdevs;
                    dataset.divRowVector(stdevs);
                }
            }

            var covarianceMatrix = dataset.transpose().mmul(dataset).divS(dataset.rows - 1);
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
            center: this.center,
            scale: this.scale,
            means: this.means,
            stdevs: this.stdevs,
            U: this.U,
            S: this.S,
        };
    }

    /**
     * Projects the dataset into new space of k dimensions.
     * @param {Matrix} dataset
     * @return {Matrix} dataset projected in the PCA space.
     */
    predict(dataset) {
        dataset = new Matrix(dataset);

        if (this.center) {
            dataset.subRowVector(this.means);
            if (this.scale) {
                dataset.divRowVector(this.stdevs);
            }
        }
        
        return dataset.mmul(this.U);
    }

    /**
     * Returns the proportion of variance for each component.
     * @return {[number]}
     */
    getExplainedVariance() {
        var sum = 0;
        for (var i = 0; i < this.S.length; i++) {
            sum += this.S[i];
        }
        return this.S.map(value => value / sum);
    }

    /**
     * Returns the cumulative proportion of variance.
     * @return {[number]}
     */
    getCumulativeVariance() {
        var explained = this.getExplainedVariance();
        for (var i = 1; i < explained.length; i++) {
            explained[i] += explained[i - 1];
        }
        return explained;
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
     * @returns {[number]}
     */
    getEigenvalues() {
        return this.S;
    }

    /**
     * Returns the standard deviations of the principal components
     * @returns {[number]}
     */
    getStandardDeviations() {
        return this.S.map(x => Math.sqrt(x));
    }

    /**
     * Returns the loadings matrix
     * @return {Matrix}
     */
    getLoadings() {
        return this.U.transpose();
    }
}

module.exports = PCA;
