import { summarize } from './services/aiService.js';

const btnSummarize = document.getElementById('btnSummarize');
const output = document.getElementById('output');
const progress = document.getElementById('progress');
const languageSelect = document.getElementById('language');

// Load saved language preference
chrome.storage.sync.get(['preferredLanguage'], (result) => {
  if (result.preferredLanguage) {
    languageSelect.value = result.preferredLanguage;
  }
});

languageSelect.addEventListener('change', () => {
  chrome.storage.sync.set({ preferredLanguage: languageSelect.value });
});

btnSummarize.addEventListener('click', () => {
  output.textContent = '';
  progress.textContent = 'Carregando texto da página...';

  // Check LanguageModel availability
  if (!('LanguageModel' in window)) {
    output.textContent = '⚠️ A API LanguageModel não está disponível. Por favor, habilite as flags experimentais do Chrome.';
    progress.textContent = '';
    return;
  }

  // Request page content from content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0].id) return;
    chrome.tabs.sendMessage(tabs[0].id, { action: 'getText' }, async (response) => {
      if (!response || !response.text) {
        output.textContent = 'Erro ao obter texto da página.';
        progress.textContent = '';
        return;
      }

      const format = document.querySelector('input[name="format"]:checked').value;
      const language = languageSelect.value;

      progress.textContent = 'Gerando resumo...';

      try {
        // Call summarize with streaming
        for await (const chunk of summarize(response.text, { language, format })) {
          output.textContent += chunk;
          progress.textContent = 'Processando...';
        }
        progress.textContent = 'Resumo completo.';
      } catch (error) {
        output.textContent = 'Erro ao gerar resumo: ' + error.message;
        progress.textContent = '';
      }
    });
  });
});
