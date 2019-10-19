import { PCA } from '../pca';

describe('PCA model', function () {
  it('Save / load model', function () {
    let dataset = [
      [2, 1, 0, 3.7],
      [3, 1, 0, 3.2],
      [2.5, 1, 0, 3.1],
      [2.1, 1, 0, 3],
    ];
    let pca = new PCA(dataset, { scale: true, ignoreZeroVariance: true });
    expect(pca.predict(dataset).rows).toStrictEqual(4);
    expect(pca.predict(dataset).columns).toStrictEqual(2);

    let model = JSON.stringify(pca.toJSON());

    let newpca = PCA.load(JSON.parse(model));
    expect(newpca.predict(dataset).rows).toStrictEqual(4);
    expect(newpca.predict(dataset).columns).toStrictEqual(2);
  });
});
