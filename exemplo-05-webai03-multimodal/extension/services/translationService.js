import { TranslationService } from "../../services/translationService.js";

const translationServiceInstance = new TranslationService();

export async function translateToPortuguese(text) {
  return translationServiceInstance.translateToPortuguese(text);
}
