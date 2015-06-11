'use strict';
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');

module.exports = function pca(dataset) {

    var featureNormalize = function(dataset) {
        var means = Stat.matrix.mean(dataset);
        var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
        means = Matrix.rowVector(means);

        var result = dataset.clone().addRowVector(means.neg());
        return result.divRowVector(std);
    };

    console.log(featureNormalize(dataset));

};
