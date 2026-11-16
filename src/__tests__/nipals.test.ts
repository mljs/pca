import { Matrix } from 'ml-matrix';
import { describe, it, expect } from 'vitest';

import { PCA } from '../pca';

describe('NIPALS variances', () => {
  it('preserves Matrix inputs and returns their projected sample variances', () => {
    const dataset = [
      [1, 2],
      [3, 3],
      [5, 7],
      [7, 8],
    ];
    const input = new Matrix(dataset);
    const pca = new PCA(input, { method: 'NIPALS' });
    const reference = new PCA(dataset, { method: 'SVD' });
    expect(input.to2DArray()).toEqual(dataset);

    const scores = pca.predict(input);
    const variances = scores.variance('column', { unbiased: true });
    expect(scores.to2DArray()).toEqual(pca.predict(dataset).to2DArray());
    expect(input.to2DArray()).toEqual(dataset);
    for (let component = 0; component < 2; component++) {
      expect(pca.getEigenvalues()[component]).toBeCloseTo(
        variances[component],
        8,
      );
      expect(variances[component]).toBeCloseTo(
        reference.getEigenvalues()[component],
        8,
      );
    }
  });

  it.each([
    { values: [1, 3], variance: 2 },
    { values: [1, 2, 3], variance: 1 },
    { values: [-3, -2, -1, 1, 2, 3], variance: 5.6 },
  ])('uses sample variance for $values', ({ values, variance }) => {
    const pca = new PCA(
      values.map((value) => [value]),
      { method: 'NIPALS', nCompNIPALS: 1 },
    );

    expect(pca.getEigenvalues()[0]).toBeCloseTo(variance, 10);
    expect(pca.getStandardDeviations()[0]).toBeCloseTo(Math.sqrt(variance), 10);
  });

  it.each(['SVD', 'covarianceMatrix'] as const)(
    'agrees with %s for a two-component dataset',
    (method) => {
      const dataset = [
        [1, 2],
        [3, 3],
        [5, 7],
        [7, 8],
      ];
      const pca = new PCA(dataset, { method: 'NIPALS' });
      const reference = new PCA(dataset, { method });

      for (let component = 0; component < 2; component++) {
        expect(pca.getEigenvalues()[component]).toBeCloseTo(
          reference.getEigenvalues()[component],
          8,
        );
        expect(pca.getStandardDeviations()[component]).toBeCloseTo(
          reference.getStandardDeviations()[component],
          8,
        );
      }
    },
  );
});
