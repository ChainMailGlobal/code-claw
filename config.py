import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # vLLM endpoints (host ports — localhost resolves on VM host)
    MISTRAL_URL: str = os.getenv("MISTRAL_URL", "http://localhost:8000")
    QWEN_URL: str = os.getenv("QWEN_URL", "http://localhost:8001")
    MISTRAL_MODEL: str = os.getenv(
        "MISTRAL_MODEL", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

    # Token Factory API (Nebius) — large model planning pass
    TOKEN_FACTORY_API_KEY: str = os.getenv("TOKEN_FACTORY_API_KEY", "")
    TOKEN_FACTORY_BASE_URL: str = os.getenv(
        "TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
    )
    # Anthropic — Claude Code CLI auth
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # MMCP — internal API auth
    MMCP_URL: str = os.getenv("MMCP_URL", "http://localhost:5000")
    MMCP_UI_TOKEN: str = os.getenv("MMCP_UI_TOKEN", "")
    # Memory isolation — owner scope for bubble tagging and MAG queries
    OWNER_ACCOUNT_ID: str = os.getenv("OWNER_ACCOUNT_ID", "chainmail")

    # Model selection for planning
    MODEL_CODING: str = os.getenv("MODEL_CODING", "deepseek-ai/DeepSeek-V3-0324")
    MODEL_REASONING: str = os.getenv("MODEL_REASONING", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1")
    MODEL_THINKING: str = os.getenv("MODEL_THINKING", "deepseek-ai/DeepSeek-R1-0528")

    # GitHub
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # TTS
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "piper")
    PIPER_MODEL_DIR: str = os.getenv("PIPER_MODEL_DIR", "/data/piper-models")
    PIPER_MODEL_NAME: str = os.getenv("PIPER_MODEL_NAME", "en_US-ryan-high")

    # Service auth
    CODE_CLAW_SECRET: str = os.getenv("CODE_CLAW_SECRET", "")
    CODE_CLAW_PORT: int = int(os.getenv("CODE_CLAW_PORT", "6000"))

    # Execution
    EXECUTOR_TIMEOUT: int = int(os.getenv("EXECUTOR_TIMEOUT", "120"))

    # Paths
    WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", "/workspace")
    DATA_DIR: str = os.getenv("DATA_DIR", "/data")

    @classmethod
    def mistral_chat_url(cls) -> str:
        return f"{cls.MISTRAL_URL}/v1/chat/completions"

    @classmethod
    def qwen_chat_url(cls) -> str:
        return f"{cls.QWEN_URL}/v1/chat/completions"

    @classmethod
    def token_factory_chat_url(cls) -> str:
        return f"{cls.TOKEN_FACTORY_BASE_URL}/chat/completions"

    @classmethod
    def active_repo_path(cls) -> str:
        return os.path.join(cls.DATA_DIR, "active_repo.json")

    @classmethod
    def piper_model_path(cls) -> str:
        return os.path.join(cls.PIPER_MODEL_DIR, f"{cls.PIPER_MODEL_NAME}.onnx")
