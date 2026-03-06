source .env

curl --silent -X POST "https://openrouter.ai/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "HTTP-Referer: https://localhost:3000" \
  -H "X-Title: My Example" \
  -d '{
        "model": '"'nvidia/nemotron-3-nano-30b-a3b:free'"', 
        "messages": [
            {"role":"user","content":"Me conte uma curiosidade sobre LLMs"}],
        "temperature": 0.3,
        "max_tokens": 1000
    }'

# {
#   "id": "gen-1772756624-L13hgtdqGex54sfZkI3V",
#   "object": "chat.completion",
#   "created": 1772756624,
#   "model": "nvidia/nemotron-3-nano-30b-a3b:free",
#   "provider": "Nvidia",
#   "choices": [
#     {
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "stop",
#       "native_finish_reason": "stop",
#       "message": {
#         "role": "assistant",
#         "content": "Claro! 🎉\n\n**Curiosidade:**  \nOs grandes modelos de linguagem (LLMs) podem “descobrir” habilidades que nem mesmo seus criadores previram, simplesmente ao escalar o tamanho do modelo e o volume de dados de treinamento. Por exemplo, ao chegar a centenas de bilhões de parâmetros, alguns LLMs começaram a demonstrar **raciocínio de cadeia de pensamento (chain‑of‑thought)** e até **resolução de problemas matemáticos** de forma emergente, embora esse tipo de competência não tenha sido explicitamente programada nem treinada diretamente. Isso sugere que, quando um modelo fica suficientemente grande, novas “capacidades” podem surgir quase como um efeito colateral da própria arquitetura e do aprendizado em escala. 🚀\n\nIsso tem levado pesquisadores a estudar “** emergent abilities**” como um fenômeno de interesse, buscando entender como e por que essas habilidades aparecem apenas acima de certos limites de tamanho ou de dados.",
#         "refusal": null,
#         "reasoning": "The user asks in Portuguese: \"Me conte uma curiosidade sobre LLMs\" (Tell me a curiosity about LLMs). We need to respond in Portuguese, providing a curiosity about large language models. Should be concise, engaging. No disallowed content. Provide a fun fact. Possibly about training data, emergent abilities, scaling laws, etc. Let's answer.\n",
#         "reasoning_details": [
#           {
#             "type": "reasoning.text",
#             "text": "The user asks in Portuguese: \"Me conte uma curiosidade sobre LLMs\" (Tell me a curiosity about LLMs). We need to respond in Portuguese, providing a curiosity about large language models. Should be concise, engaging. No disallowed content. Provide a fun fact. Possibly about training data, emergent abilities, scaling laws, etc. Let's answer.\n",
#             "format": "unknown",
#             "index": 0
#           }
#         ]
#       }
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 24,
#     "completion_tokens": 297,
#     "total_tokens": 321,
#     "cost": 0,
#     "is_byok": false,
#     "prompt_tokens_details": {
#       "cached_tokens": 0,
#       "cache_write_tokens": 0,
#       "audio_tokens": 0,
#       "video_tokens": 0
#     },
#     "cost_details": {
#       "upstream_inference_cost": 0,
#       "upstream_inference_prompt_cost": 0,
#       "upstream_inference_completions_cost": 0
#     },
#     "completion_tokens_details": {
#       "reasoning_tokens": 86,
#       "image_tokens": 0,
#       "audio_tokens": 0
#     }
#   }
# }