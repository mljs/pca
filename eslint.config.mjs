import cheminfo from 'eslint-config-cheminfo-typescript';
import globals from 'globals';

export default [
  ...cheminfo,
  {
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
    rules: {}
  }
]