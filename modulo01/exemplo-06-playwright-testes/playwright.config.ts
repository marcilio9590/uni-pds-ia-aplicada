// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  use: {
    baseURL: 'https://erickwendel.github.io/vanilla-js-web-app-example/',
    timeout: 5000, // 5 seconds
  },
});
