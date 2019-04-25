import { Matrix, MatrixTransposeView, EVD, SVD } from 'ml-matrix';

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} dataset - dataset or covariance matrix
 * @param {Object} [options]
 * @param {boolean} [options.isCovarianceMatrix=false] - true if the dataset is a covariance matrix
 * @param {boolean} [options.useCovarianceMatrix=false] - force the use of the covariance matrix instead of singular value decomposition.
 * @param {boolean} [options.center=true] - should the data be centered (subtract the mean)
 * @param {boolean} [options.scale=false] - should the data be scaled (divide by the standard deviation)
 * */
export class PCA {
  constructor(dataset, options = {}) {
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

    dataset = new Matrix(dataset);

    const {
      isCovarianceMatrix = false,
      center = true,
      scale = false
    } = options;

    this.center = center;
    this.scale = scale;
    this.means = null;
    this.stdevs = null;

    if (isCovarianceMatrix) {
      // user provided a covariance matrix instead of dataset
      this._computeFromCovarianceMatrix(dataset);
      return;
    }

    var useCovarianceMatrix;
    if (typeof options.useCovarianceMatrix === 'boolean') {
      useCovarianceMatrix = options.useCovarianceMatrix;
    } else {
      useCovarianceMatrix = dataset.rows > dataset.columns;
    }

    if (useCovarianceMatrix) {
      // user provided a dataset but wants us to compute and use the covariance matrix
      this._adjust(dataset);
      const covarianceMatrix = new MatrixTransposeView(dataset)
        .mmul(dataset)
        .div(dataset.rows - 1);
      this._computeFromCovarianceMatrix(covarianceMatrix);
    } else {
      this._adjust(dataset);
      var svd = new SVD(dataset, {
        computeLeftSingularVectors: false,
        computeRightSingularVectors: true,
        autoTranspose: true
      });

      this.U = svd.rightSingularVectors;

      const singularValues = svd.diagonal;
      const eigenvalues = [];
      for (const singularValue of singularValues) {
        eigenvalues.push((singularValue * singularValue) / (dataset.rows - 1));
      }
      this.S = eigenvalues;
    }
  }

  /**
   * Load a PCA model from JSON
   * @param {Object} model
   * @return {PCA}
   */
  static load(model) {
    if (typeof model.name !== 'string') {
      throw new TypeError('model must have a name property');
    }
    if (model.name !== 'PCA') {
      throw new RangeError(`invalid model: ${model.name}`);
    }
    return new PCA(true, model);
  }

  /**
   * Project the dataset into the PCA space
   * @param {Matrix} dataset
   * @param {Object} options
   * @return {Matrix} dataset projected in the PCA space
   */
  predict(dataset, options = {}) {
    const { nComponents = this.U.columns } = options;

    dataset = new Matrix(dataset);
    if (this.center) {
      dataset.subRowVector(this.means);
      if (this.scale) {
        dataset.divRowVector(this.stdevs);
      }
    }

    var predictions = dataset.mmul(this.U);
    return predictions.subMatrix(0, predictions.rows - 1, 0, nComponents - 1);
  }

  /**
   * Returns the proportion of variance for each component
   * @return {[number]}
   */
  getExplainedVariance() {
    var sum = 0;
    for (const s of this.S) {
      sum += s;
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
   * @returns {Matrix}
   */
  getEigenvectors() {
    return this.U;
  }

  /**
   * Returns the Eigenvalues (on the diagonal)
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
      S: this.S
    };
  }

  _adjust(dataset) {
    if (this.center) {
      const mean = dataset.mean('column');
      const stdevs = this.scale
        ? dataset.standardDeviation('column', { mean })
        : null;
      this.means = mean;
      dataset.subRowVector(mean);
      if (this.scale) {
        for (var i = 0; i < stdevs.length; i++) {
          if (stdevs[i] === 0) {
            throw new RangeError(
              `Cannot scale the dataset (standard deviation is zero at index ${i}`
            );
          }
        }
        this.stdevs = stdevs;
        dataset.divRowVector(stdevs);
      }
    }
  }

  _computeFromCovarianceMatrix(dataset) {
    const evd = new EVD(dataset, { assumeSymmetric: true });
    this.U = evd.eigenvectorMatrix;
    this.U.flipRows();
    this.S = evd.realEigenvalues;
    this.S.reverse();
  }
}
