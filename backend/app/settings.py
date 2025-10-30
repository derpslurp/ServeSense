from pydantic import BaseSettings


class Settings(BaseSettings):
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 100
    CHUNK_SIZE_BYTES: int = 1024 * 1024  # 1 MB

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()


