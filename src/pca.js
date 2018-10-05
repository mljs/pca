
const matrixLib = require('ml-matrix');

const Matrix = matrixLib.Matrix;
const EVD = matrixLib.EVD;
const SVD = matrixLib.SVD;
const Stat = require('ml-stat/matrix');

const mean = Stat.mean;
const stdev = Stat.standardDeviation;

const defaultOptions = {
  isCovarianceMatrix: false,
  center: true,
  scale: false,
  algorithm: 'svd'
};

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} dataset - dataset or covariance matrix
 * @param {object} options
 * @param {boolean} [options.isCovarianceMatrix=false] - true if the dataset is a covariance matrix
 * @param {boolean} [options.center=true] - should the data be centered (subtract the mean)
 * @param {boolean} [options.scale=false] - should the data be scaled (divide by the standard deviation)
 * */
class PCA {
  constructor(dataset, options) {
    if (dataset === true) {
      const model = options;
      this.center = model.center;
      this.scale = model.scale;
      this.means = model.means;
      this.stdevs = model.stdevs;
      this.U = Matrix.checkMatrix(model.U);
      this.S = model.S;
      return;
    }

    options = Object.assign({}, defaultOptions, options);
    this.center = false;
    this.scale = false;
    this.means = null;
    this.stdevs = null;


    if (options.algorithm.toLowerCase() === 'nipals') {
      this._nipals(dataset, options);
    } else {
      this._svd(dataset, options);
    }
  }

  /**
     * Load a PCA model from JSON
     * @param {object} model
     * @return {PCA}
     */
  static load(model) {
    if (model.name !== 'PCA') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }
    return new PCA(true, model);
  }

  /**
     * Project the dataset into the PCA space
     * @param {Matrix} dataset
     * @param {object} options
     * @return {Matrix} dataset projected in the PCA space
     */
  predict(dataset, options = {}) {
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
     * Returns the proportion of variance for each component
     * @return {[number]}
     */
  getExplainedVariance() {
    var sum = 0;
    for (var i = 0; i < this.S.length; i++) {
      sum += this.S[i];
    }
    return this.S.map((value) => value / sum);
  }

  /**
     * Returns the cumulative proportion of variance
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
     * Returns the Eigenvectors of the covariance matrix
     * @return {Matrix}
     */
  getEigenvectors() {
    return this.U;
  }

  /**
     * Returns the Eigenvalues (on the diagonal)
     * @return {[number]}
     */
  getEigenvalues() {
    return this.S;
  }

  /**
     * Returns the standard deviations of the principal components
     * @return {[number]}
     */
  getStandardDeviations() {
    return this.S.map((x) => Math.sqrt(x));
  }

  /**
     * Returns the loadings matrix
     * @return {Matrix}
     */
  getLoadings() {
    return this.U.transpose();
  }

  /**
     * Export the current model to a JSON object
     * @return {object} model
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

  _svd(dataset, options) {
    if (options.isCovarianceMatrix) { // user provided a covariance matrix instead of dataset
      this._computeFromCovarianceMatrix(dataset);
      return;
    }
    var useCovarianceMatrix;
    if (typeof options.useCovarianceMatrix === 'boolean') {
      useCovarianceMatrix = options.useCovarianceMatrix;
    } else {
      useCovarianceMatrix = dataset.length > dataset[0].length;
    }
    dataset = this._adjust(dataset, options);

    if (useCovarianceMatrix) { // user provided a dataset but wants us to compute and use the covariance matrix
      const covarianceMatrix = dataset.transposeView().mmul(dataset).div(dataset.rows - 1);
      this._computeFromCovarianceMatrix(covarianceMatrix);
    } else {
      var svd = new SVD(dataset, {
        computeLeftSingularVectors: false,
        computeRightSingularVectors: true,
        autoTranspose: true
      });

      this.U = svd.rightSingularVectors;

      const singularValues = svd.diagonal;
      const eigenvalues = new Array(singularValues.length);
      for (var i = 0; i < singularValues.length; i++) {
        eigenvalues[i] = singularValues[i] * singularValues[i] / (dataset.length - 1);
      }
      this.S = eigenvalues;
    }
  }

  _nipals(dataset, options = {}) {
    dataset = this._adjust(dataset, options);
    var {
      nComponents = 2,
      threshold = 1e-9,
      maxIterations = 100
    } = options;

    var eMatrix = this._adjust(dataset, options);

    var r = eMatrix.rows;
    var c = eMatrix.columns;

    var T = Matrix.zeros(r, nComponents);
    var P = Matrix.zeros(c, nComponents);
    var eigenvalues = new Array(nComponents);

    for (let i = 0; i < nComponents; i++) {
      let tIndex = maxSumColIndex(eMatrix.clone().mulM(eMatrix));
      let t = eMatrix.getColumnVector(tIndex);

      let k = 0;
      let tNew = t.dot(t);
      for (let tOld = Number.MAX_SAFE_INTEGER; Math.abs(tNew - tOld) > threshold && k < maxIterations; k++) {
        var p = getLoading(eMatrix, t);
        t = eMatrix.mmul(p);
        tOld = tNew;
        tNew = t.dot(t);
      }
      eigenvalues[i] = tNew;
      T.setColumn(i, t);
      P.setColumn(i, p);
      eMatrix.sub(t.mmul(p.transpose()));
    }
    this.U = P.transpose();
    this.T = T;
    this.S = eigenvalues;
  }

  _adjust(dataset, options) {
    this.center = !!options.center;
    this.scale = !!options.scale;

    dataset = new Matrix(dataset);

    if (this.center) {
      const means = mean(dataset);
      const stdevs = this.scale ? stdev(dataset, means, true) : null;
      this.means = means;
      dataset.subRowVector(means);
      if (this.scale) {
        for (var i = 0; i < stdevs.length; i++) {
          if (stdevs[i] === 0) {
            throw new RangeError(`Cannot scale the dataset (standard deviation is zero at index ${i}`);
          }
        }
        this.stdevs = stdevs;
        dataset.divRowVector(stdevs);
      }
    }

    return dataset;
  }

  _computeFromCovarianceMatrix(dataset) {
    const evd = new EVD(dataset, { assumeSymmetric: true });
    this.U = evd.eigenvectorMatrix;
    for (var i = 0; i < this.U.length; i++) {
      this.U[i].reverse();
    }
    this.S = evd.realEigenvalues.reverse();
  }
}

function getLoading(e, t) {
  var m = e.columns;
  var n = e.rows;

  var result = new Matrix(m, 1);

  var Bcolj = new Array(n);
  for (let i = 0; i < m; i++) {
    var s = 0;
    for (let k = 0; k < n; k++) {
      s += e.get(k, i) * t[k][0];
    }
    result.set(i, 0, s);
  }
  return result.mul(1 / result.norm());
}

/**
 * @private
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} data
 * @return {number} index of the maximum
 */
function maxSumColIndex(data) {
  return data.sum('column').maxIndex()[0];
}
module.exports = PCA;
