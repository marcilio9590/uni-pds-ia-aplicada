import { test, expect } from "@playwright/test";

test.describe("form submission and validation", () => {
  test("submits the form and adds a new item to the list", async ({ page }) => {
    await page.goto("/vanilla-js-web-app-example/");

    const title = `Playwright Test Title ${Date.now()}`;
    const url = "https://www.shutterstock.com/image-photo/traveler-woman-arms-raised-triumph-260nw-2457990309.jpg";

    // Count headings before
    const headingsBefore = await page.getByRole("heading").count();

    // Fill form
    await page.getByRole("textbox", { name: "Image Title" }).fill(title);
    await page.getByRole("textbox", { name: "Image URL" }).fill(url);

    // Submit
    await page.getByRole("button", { name: "Submit Form" }).click();

    // Expect a heading with the new title to appear and total headings to increase
    await page
      .getByRole("heading", { name: title })
      .waitFor({ state: "visible" });
    const headingsAfter = await page.getByRole("heading").count();
    expect(headingsAfter).toBeGreaterThan(headingsBefore);
  });

  test("validates form: missing title or invalid url should not add item", async ({
    page,
  }) => {
    await page.goto("/vanilla-js-web-app-example/");

    const goodUrl = "https://www.shutterstock.com/image-photo/traveler-woman-arms-raised-triumph-260nw-2457990309.jpg";
    const badUrl = "not-a-valid-url";

    // baseline count
    const baseline = await page.getByRole("heading").count();

    // Case 1: missing title
    await page.getByRole("textbox", { name: "Image Title" }).fill("");
    await page.getByRole("textbox", { name: "Image URL" }).fill(goodUrl);
    await page.getByRole("button", { name: "Submit Form" }).click();
    // ensure no new heading was added
    const afterMissingTitle = await page.getByRole("heading").count();
    expect(afterMissingTitle).toBe(baseline);

    // Case 2: invalid URL
    await page
      .getByRole("textbox", { name: "Image Title" })
      .fill("Invalid URL Test");
    await page.getByRole("textbox", { name: "Image URL" }).fill(badUrl);
    await page.getByRole("button", { name: "Submit Form" }).click();
    const afterInvalidUrl = await page.getByRole("heading").count();
    expect(afterInvalidUrl).toBe(baseline);
  });
});
