import { Matrix } from 'ml-matrix';

declare module 'ml-pca' {
  export interface IPCAOptions {
    isCovarianceMatrix?: boolean;
    method?: 'SVD' | 'NIPALS' | 'covarianceMatrix';
    center?: boolean;
    scale?: boolean;
    nCompNIPALS?: number;
    ignoreZeroVariance?: boolean;
  }

  export interface IPCAModel {
    name: 'PCA';
  }

  export interface IPredictOptions {
    nComponents?: number;
  }

  export class PCA {
    constructor(dataset: number[][] | Matrix, options?: IPCAOptions);
    static load(model: IPCAModel): PCA;
    predict(dataset: number[][] | Matrix, options?: IPredictOptions): Matrix;
    invert(dataset: number[][] | Matrix): Matrix;
    getExplainedVariance(): number[];
    getCumulativeVariance(): number[];
    getEigenvectors(): Matrix;
    getEigenvalues(): number[];
    getStandardDeviations(): number[];
    getLoadings(): Matrix;
    toJSON(): IPCAModel;
  }
}
