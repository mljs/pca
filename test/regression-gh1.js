'use strict';

var PCA = require('..');

// See https://github.com/mljs/pca/issues/1
describe.only('Regression test for special datasets', function () {
    it('should work with one element', function () {
        new PCA([[1,0],[2,0]]);
    });
});

//
// Matlab http://ch.mathworks.com/help/stats/pca.html
// pca([[1,0],[2,0]])

/*

 u =

 1 0
 2 0

 [coeff, score, latent, tsquared, explained, mu]=pca(u)

 coeff =

 1
 0


 score =

 -0.5000
 0.5000


 latent =

 0.5000


 tsquared =

 0
 0


 explained =

 100


 mu =

 1.5000 0


 */