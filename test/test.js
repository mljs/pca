'use strict';
var pca = require('../src/pca');
var Matrix = require('ml-matrix');

describe('PCA algorithm', function () {
    it('Main test', function () {
        pca(Matrix([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10]]));
    })
});