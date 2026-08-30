from .package import build_submission_archive

__all__ = ["OUTPUT_PATTERN", "build_submission_archive", "validate_submission_frame"]


def __getattr__(name: str):
    if name in {"OUTPUT_PATTERN", "validate_submission_frame"}:
        from .contract import OUTPUT_PATTERN, validate_submission_frame

        return {
            "OUTPUT_PATTERN": OUTPUT_PATTERN,
            "validate_submission_frame": validate_submission_frame,
        }[name]
    raise AttributeError(name)
