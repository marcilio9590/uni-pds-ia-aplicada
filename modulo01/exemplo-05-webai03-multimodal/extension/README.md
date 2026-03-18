# Resumidor AI Chrome Extension

## Instruções para carregar a extensão

1. Abra o Chrome e vá para `chrome://extensions/`
2. Ative o modo desenvolvedor no canto superior direito
3. Clique em "Carregar sem empacotar"
4. Selecione a pasta `exemplo-05-webai03-multimodal/extension`

## Checklist de testes manuais

- Abrir uma página de artigo longa
- Clicar no ícone da extensão na barra do Chrome
- Selecionar o idioma de saída (Português/Inglês)
- Selecionar o formato do resumo (5 linhas, pontos principais, lista de ações)
- Clicar no botão "Resumo"
- Verificar que o texto da página (ou seleção) é extraído e enviado para o popup
- Verificar que o resumo é gerado e exibido progressivamente
- Se a API LanguageModel não estiver disponível, verificar a mensagem de instrução
- Verificar que a preferência de idioma é salva e recarregada

## Observações

- Dependência das flags experimentais do Chrome para LanguageModel, Translator e LanguageDetector
- Streaming do resumo para melhor UX
- Sem backend, tudo roda localmente na extensão
