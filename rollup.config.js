export default {
  input: 'src/pca.js',
  output: {
    file: 'pca.js',
    format: 'cjs',
    exports: 'named'
  },
  external: ['ml-matrix']
};
