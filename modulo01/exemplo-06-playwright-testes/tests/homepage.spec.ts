import { test, expect } from "@playwright/test";

test("homepage loads and shows title", async ({ page }) => {
  await page.goto("/vanilla-js-web-app-example/");
  await expect(page).toHaveTitle(/TDD Frontend Example/);
});

test("form and articles are present", async ({ page }) => {
  await page.goto("/vanilla-js-web-app-example/");
  // Check for form fields
  await expect(
    page.getByRole("textbox", { name: "Image Title" }),
  ).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Image URL" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Submit Form" })).toBeVisible();
  // Check for articles
  await expect(page.getByRole("heading", { name: "AI Alien" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Predator Night Vision" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "ET Bilu" })).toBeVisible();
});
