'use strict';
var Matrix = require('ml-matrix');

module.exports = function pca(dataset) {
    var powArray = function(i, j) {
        this[i][j] = Math.pow(this[i][j], 2);
        return this;
    };

    var mean = function(data) {
        return data.sum() / data[0].length;
    };

    var standardDeviation = function(data) {
        var dataMean = mean(data);

        var result = data.clone().add(-dataMean);
        result.apply(powArray);
        result = result.sum() / (data[0].length - 1);

        return Math.sqrt(result);
    };

    var featureNormalize = function(dataset) {

    };

};
