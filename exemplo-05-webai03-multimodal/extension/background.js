// background.js
// Currently optional: can be used for message routing or future control logic

chrome.runtime.onInstalled.addListener(() => {
  console.log('Resumidor AI extension installed');
});
