from dataclasses import dataclass, field


@dataclass()
class StreamLoggingParams:
    config_path: str = field(default="configs/loggers/logger_config.yaml")
    logger: str = field(default="stream_logger")
    field_splitter: str = field(default="\n====================\n")

