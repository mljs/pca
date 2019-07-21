import { PCA } from '../pca';

describe('PCA algorithm', function () {
  var testDataset = [
    [3.38156266663556, 3.38911268489207],
    [4.52787538040321, 5.85417810116941],
    [2.65568186873946, 4.41199471748479],
    [2.76523467422508, 3.71541364974329],
    [2.84656010622109, 4.17550644951439],
    [3.89067195630921, 6.48838087188621],
    [3.47580524144079, 3.63284876204706],
    [5.91129844549583, 6.68076852676779],
    [3.92889396796927, 5.09844660814783],
    [4.56183536608942, 5.62329929038287],
    [4.57407170552516, 5.39765068914995],
    [4.37173355733069, 5.46116548918004],
    [4.191693876251, 4.95469359045186],
    [5.24408517686664, 4.66148766849075],
    [2.83584020280787, 3.76801716326883],
    [5.63526969258877, 6.3121143831056],
    [4.68632967964966, 5.66524110304899],
    [2.85051337486241, 4.62645627270763],
    [5.11015730037567, 7.36319662353662],
    [5.18256376844695, 4.64650908778182],
    [5.70732809135459, 6.68103994977504],
    [3.57968458251575, 4.80278073546266],
    [5.63937773123337, 6.12043594486419],
    [4.2634685116016, 4.68942896498378],
    [2.5365169312575, 3.88449077575653],
    [3.22382901750257, 4.94255585367287],
    [4.92948801055806, 5.95501971122402],
    [5.79295773976472, 5.10839305453511],
    [2.81684823843681, 4.81895768959782],
    [3.88882413905485, 5.10036563684974],
    [3.34323419214569, 5.89301345482551],
    [5.87973413931621, 5.52141663871971],
    [3.10391912309722, 3.85710242154672],
    [5.33150572016357, 4.68074234658945],
    [3.37542686902548, 4.56537851617577],
    [4.77667888193414, 6.25435038973932],
    [2.67574630193237, 3.73096987540176],
    [5.50027665196111, 5.67948113445839],
    [1.79709714108619, 3.24753885348582],
    [4.32251470267314, 5.11110472186451],
    [4.42100444798251, 6.02563977712186],
    [3.1792988626619, 4.43686031619158],
    [3.03354124664264, 3.97879278223097],
    [4.60934820070329, 5.87979200261535],
    [2.96378859260761, 3.30024834860712],
    [3.97176248181608, 5.40773735417849],
    [1.18023320575165, 2.87869409391385],
    [1.91895045046187, 5.07107847507096],
    [3.95524687147485, 4.50532709674253],
    [5.11795499426461, 6.08507386392396],
  ];

  var pca = new PCA(testDataset, {
    scale: true,
  });

  it('PCA Main test', function () {
    var U = [[0.7071, 0.7071], [0.7071, -0.7071]];
    var S = [1.73553, 0.2644696];

    var currentU = pca.getEigenvectors();
    var currentS = pca.getEigenvalues();

    for (var i = 0; i < 2; ++i) {
      for (var j = 0; j < 2; ++j) {
        expect(currentU.get(i, j)).toBeCloseTo(U[i][j], 3);
      }
    }

    for (i = 0; i < 2; ++i) {
      expect(currentS[i]).toBeCloseTo(S[i], 3);
    }
  });

  it('Projection method', function () {
    var result = pca.predict(testDataset, 1);
    expect(result.get(0, 0)).toBeCloseTo(-1.481274, 5);
  });

  it('Variance explained method', function () {
    var varianceExplained = pca.getExplainedVariance();
    expect(varianceExplained[0]).toBeCloseTo(0.8678, 4);
    expect(varianceExplained[1]).toBeCloseTo(0.1322, 4);
  });

  it('Export and import', function () {
    var model = JSON.stringify(pca.toJSON());
    var newpca = PCA.load(JSON.parse(model));

    var U = [[0.7071, 0.7071], [0.7071, -0.7071]];
    var S = [1.73553, 0.2644696];

    var currentU = newpca.getEigenvectors();
    var currentS = newpca.getEigenvalues();

    for (var i = 0; i < 2; ++i) {
      for (var j = 0; j < 2; ++j) {
        expect(currentU.get(i, j)).toBeCloseTo(U[i][j], 3);
      }
    }

    for (i = 0; i < 2; ++i) {
      expect(currentS[i]).toBeCloseTo(S[i], 3);
    }
  });

  it('Standardization error with constant column', function () {
    let dataset = [[1, 2, 3.7], [1, 3, 3.2], [1, 2.5, 3.1], [1, 2.1, 3]];
    expect(() => new PCA(dataset, { scale: true })).toThrow(
      /standard deviation is zero at index 0/,
    );
  });

  it('Standardization error with constant column - ignoreZeroVariance', function () {
    let dataset = [
      [2, 1, 0, 3.7],
      [3, 1, 0, 3.2],
      [2.5, 1, 0, 3.1],
      [2.1, 1, 0, 3],
    ];
    let newpca = new PCA(dataset, { scale: true, ignoreZeroVariance: true });
    expect(newpca.getLoadings().rows).toBe(2);
    expect(newpca.predict(dataset).rows).toStrictEqual(4);
    expect(newpca.predict(dataset).columns).toStrictEqual(2);
  });

  it('Test number components in function predict', function () {
    var dataset = [[1, 2, 0], [3, 4, 0], [5, 6, 0]];
    var newpca = new PCA(dataset);
    expect(newpca.predict(dataset, { nComponents: 2 }).columns).toBe(2);
  });

  it('should throw on load if wrong model', () => {
    expect(() => PCA.load({})).toThrow(/model must have a name property/);
    expect(() => PCA.load({ name: 'test' })).toThrow(/invalid model: test/);
  });

  it('should throw on wrong method', () => {
    expect(() => new PCA([[0, 1], [1, 0]], { method: 'XXX ' })).toThrow(/unknown method: XXX/);
  });
});
