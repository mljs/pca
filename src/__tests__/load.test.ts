import { describe, it, expect } from 'vitest';

import { PCA } from '../pca';

describe('PCA model', () => {
  it('Save / load model', () => {
    const dataset = [
      [2, 1, 0, 3.7],
      [3, 1, 0, 3.2],
      [2.5, 1, 0, 3.1],
      [2.1, 1, 0, 3],
    ];
    const pca = new PCA(dataset, { scale: true, ignoreZeroVariance: true });
    expect(pca.predict(dataset).rows).toBe(4);
    expect(pca.predict(dataset).columns).toBe(2);

    const model = JSON.stringify(pca.toJSON());

    const newpca = PCA.load(JSON.parse(model));
    expect(newpca.predict(dataset).rows).toBe(4);
    expect(newpca.predict(dataset).columns).toBe(2);
  });
});
