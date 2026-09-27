import {defineConfig, devices} from "@playwright/test";

const browserPort = process.env.PLAYWRIGHT_PORT || "4173";
const python = process.platform === "win32" ? "python" : "python3";

export default defineConfig({
  testDir: "./tests_browser",
  testMatch: ["management-panel.spec.mjs", "management-crud.spec.mjs", "timezone-locale-boundaries.spec.mjs"],
  grep: /renders the shipped Guide and responds to real browser interactions|general configuration survives a fresh panel load and a rejected save can be retried|persistent memories support create, reload, edit, and delete|Usage sends HA-local dates and renders the same instant in HA time|Quiet Hours saves a wall-clock time without browser timezone conversion/,
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: {
    timeout: 7_500,
  },
  reporter: process.env.CI
    ? [["line"], ["html", {outputFolder: "playwright-report", open: "never"}]]
    : "list",
  use: {
    baseURL: `http://127.0.0.1:${browserPort}`,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `${python} ci/browser_fixture_server.py ${browserPort}`,
    url: `http://127.0.0.1:${browserPort}/tests_browser/fixture.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: {...devices["Desktop Chrome"]},
    },
    {
      name: "firefox",
      use: {...devices["Desktop Firefox"]},
    },
    {
      name: "webkit",
      use: {...devices["Desktop Safari"]},
    },
  ],
});
