'use strict';
var pca = require('../src/pca');
var Matrix = require('ml-matrix');

describe('PCA algorithm', function () {
    it('Main test', function () {
        pca(Matrix.rowVector([1, 2, 3, 4, 5]));
    })
});