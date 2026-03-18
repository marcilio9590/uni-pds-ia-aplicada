// content_script.js

function extractText() {
  // Basic heuristic to exclude nav, menus, footers could be added here
  // For now, just return document.body.innerText
  return window.getSelection().toString() || document.body.innerText;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getText') {
    const text = extractText();
    sendResponse({ text });
  }
  // Return true to indicate async response if needed
  return true;
});
