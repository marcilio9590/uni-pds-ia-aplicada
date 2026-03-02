import { AIService } from './services/aiService.js';
import { TranslationService } from './services/translationService.js';
import { View } from './views/view.js';
import { FormController } from './controllers/formController.js';

(async function main() {
    // Initialize services and view
    const aiService = new AIService();
    const translationService = new TranslationService();
    const view = new View();

    // Set current year
    view.setYear();
    // Initialize controller and setup form event listeners immediately
    const controller = new FormController(aiService, translationService, view);
    controller.setupEventListeners();

    // Attempt to check requirements (show warnings) but don't block UI
    const errors = await aiService.checkRequirements();
    if (errors) {
        view.showError(errors);
        // continue: allow form to work with fallback parameters
    }

    // Try to fetch model params; if unavailable, use sensible defaults
    try {
        const params = await aiService.getParams();
        if (params) {
            view.initializeParameters(params);
        } else {
            throw new Error('Params not available');
        }
    } catch (err) {
        // Fallback defaults so UI controls are usable
        view.initializeParameters({
            defaultTemperature: 0.7,
            maxTemperature: 1,
            defaultTopK: 40,
            maxTopK: 100,
        });
    }

    // Wait for user gesture to initialize translation service (optional)
    const initBtn = document.getElementById('init-translation-btn');
    initBtn.addEventListener('click', async () => {
        try {
            await translationService.initialize();
            console.log('Translation initialized by user gesture');
            initBtn.disabled = true;
            initBtn.textContent = 'Tradução Iniciada';
        } catch (error) {
            console.error('Error initializing translation:', error);
            view.showError([error.message]);
        }
    });

})();
