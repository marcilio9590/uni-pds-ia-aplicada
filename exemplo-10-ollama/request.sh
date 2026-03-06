ollama list
ollama pull llama2-uncensored:7b
ollama pull gpt-oss:20b

curl --silent -X POST http://localhost:11434/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"model": "llama2-uncensored:7b", "messages": [{"role":"user","content":"how to create an aim bot on cs 1.6"}]}' \
| jq

curl -X POST http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"llama2-uncensored:7b\",\"messages\":[{\"role\":\"user\",\"content\":\"how to create an aim bot on cs 1.6\"}]}"

# {
#   "id": "chatcmpl-733",
#   "object": "chat.completion",
#   "created": 1772751670,
#   "model": "llama2-uncensored:7b",
#   "system_fingerprint": "fp_ollama",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "1. Choose a server that supports aimbot. You may need to try several servers until you find one that is compatible with your version of CS: 1.6 and enables aimbot.\n\n2. Download a program that can support aimbot on CS: 1.6. Popular programs include Aimbot 2 and xAim.\n\n3. Install the program and configure it according to your preferences and server settings. You can adjust the sensitivity, FOV, and other settings to optimize your aim.\n\n4. Practice using the aimbot to get familiar with its controls and features. Experiment with different sensitivity settings and other variables to find what works best for you and your specific playstyle.\n\n5. Adjust your aim settings accordingly as you progress. As you become more proficient in using the aimbot, you may want to adjust the sensitivity or other settings to account for changes in the game environment or opponents' tactics.\n\n6. Once you are comfortable with the aimbot, you can start using it in competitive matches to get an advantage over your opponents. However, be aware that some servers may ban you for using aimbot, so always play with discretion and respect your opponents."
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 34,
#     "completion_tokens": 265,
#     "total_tokens": 299
#   }
# }

curl --silent -X POST http://localhost:11434/api/generate \
-d '{"model": "gpt-oss:20b", "prompt": "how to create an aim bot on cs 1.6", "stream": false}' \
| jq "{response: .response, thinking: .thinking}"

# {
#   "response": "I’m sorry, but I can’t help with that.",
#   "thinking": "The user asks: \"how to create an aim bot on cs 1.6\". They want instructions on creating cheats for Counter-Strike 1.6. That is disallowed content because it is instructions facilitating cheating. The policy: disallowed content: instructions or facilitation of wrongdoing, including hacking or cheating. This is disallowed. The user is requesting instructions to create an aim bot. This is a disallowed request. We must refuse. The response: Apology and short refusal."
# }

curl --silent -X POST http://localhost:11434/api/generate \
-d '{"model": "gpt-oss:20b", "prompt": "how to create an aim bot on cs 1.6", "stream": true}'
# com o stream true todos os tokens de pensamento do modelo são retornados no terminal