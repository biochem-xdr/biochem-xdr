MCQ_MODELS = {
    "claude-sonnet-4.6": {
        "display":  "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "platform": "Anthropic API",
        "model_id": "claude-sonnet-4-6",
        "local":    False,
    },
    "gemini-2.5-flash": {
        "display":  "Gemini 2.5 Flash",
        "provider": "Google",
        "platform": "Google API",
        "model_id": "gemini-2.5-flash-preview",
        "local":    False,
    },
    "qwen3-235b": {
        "display":  "Qwen3-235B",
        "provider": "Alibaba",
        "platform": "Together.ai",
        "model_id": "Qwen/Qwen3-235B",
        "local":    False,
    },
    "gpt-4o": {
        "display":  "GPT-4o",
        "provider": "OpenAI",
        "platform": "OpenAI API",
        "model_id": "gpt-4o",
        "local":    False,
    },
    "deepseek-v3": {
        "display":  "DeepSeek-V3",
        "provider": "DeepSeek",
        "platform": "DeepSeek API",
        "model_id": "deepseek-chat",
        "local":    False,
    },
    "qwen3-32b": {
        "display":  "Qwen3-32B",
        "provider": "Alibaba",
        "platform": "Local (HF)",
        "model_id": "Qwen/Qwen3-32B",
        "local":    True,
    },
    "llama-3.3-70b": {
        "display":  "Llama-3.3-70B",
        "provider": "Meta",
        "platform": "Together.ai",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "local":    False,
    },
    "mistral-7b-instruct": {
        "display":  "Mistral-7B-Instruct",
        "provider": "Mistral AI",
        "platform": "Local (HF)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "local":    True,
    },
    "gemma-3-27b": {
        "display":  "Gemma-3-27B",
        "provider": "Google",
        "platform": "Local (HF)",
        "model_id": "google/gemma-3-27b-it",
        "local":    True,
    },
    "llama-3-8b-instruct": {
        "display":  "Llama-3-8B-Instruct",
        "provider": "Meta",
        "platform": "Local (HF)",
        "model_id": "meta-llama/Llama-3-8B-Instruct-Turbo",
        "local":    True,
    },
    "openbiollm-8b": {
        "display":  "OpenBioLLM-8B",
        "provider": "Saama AI",
        "platform": "Local (HF)",
        "model_id": "aaditya/Llama3-OpenBioLLM-8B",
        "local":    True,
    },
    "biomistral-7b": {
        "display":  "BioMistral-7B",
        "provider": "BioMistral",
        "platform": "Local (HF)",
        "model_id": "BioMistral/BioMistral-7B",
        "local":    True,
    },
    "medgemma-27b": {
        "display":  "MedGemma-27B",
        "provider": "Google",
        "platform": "Local (HF)",
        "model_id": "google/medgemma-27b-it",
        "local":    True,
    },
}

REASONING_MODELS = {
    "o3": {
        "display":  "OpenAI o3",
        "provider": "OpenAI",
        "platform": "OpenAI API",
        "model_id": "o3",
        "local":    False,
    },
    "deepseek-v4-pro": {
        "display":  "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "platform": "DeepSeek API",
        "model_id": "deepseek-reasoner",
        "local":    False,
    },
    "qwq-32b": {
        "display":  "QwQ-32B",
        "provider": "Alibaba",
        "platform": "Local (HF)",
        "model_id": "Qwen/QwQ-32B",
        "local":    True,
    },
    "r1-distill-qwen-32b": {
        "display":  "R1-Distill-Qwen-32B",
        "provider": "DeepSeek",
        "platform": "Local (HF)",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "local":    True,
    },
}

JUDGE_MODEL = {
    "model_id":    "gpt-4o-mini",
    "provider":    "OpenAI",
    "platform":    "OpenAI API",
    "max_tokens":  5,
    "temperature": 0,
}

GENERATION_MODEL = {
    "model_id":              "claude-sonnet-4-6",
    "provider":              "Anthropic",
    "platform":              "Anthropic API",
    "temperature_generation": 0.7,
    "temperature_regen":      0,
}

COOC_MODELS = [
    {
        "name":     "gpt-4o",
        "model_id": "gpt-4o",
        "provider": "openai",
        "category": "frontier",
    },
    {
        "name":     "claude-sonnet-4.6",
        "model_id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "category": "frontier",
    },
    {
        "name":     "llama-3.3-70b",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "provider": "together",
        "category": "general",
    },
    {
        "name":     "llama-3-8b-instruct",
        "model_id": "meta-llama/Llama-3-8B-Instruct-Turbo",
        "provider": "together",
        "category": "general",
    },
]
ALL_MCQ_MODEL_IDS = list(MCQ_MODELS.keys())