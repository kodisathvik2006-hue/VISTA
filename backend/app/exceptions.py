from fastapi import HTTPException, status


class AppError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message})
        self.code = code
        self.message = message


def bad_request(code: str, message: str) -> AppError:
    return AppError(status.HTTP_400_BAD_REQUEST, code, message)


def not_found(code: str, message: str) -> AppError:
    return AppError(status.HTTP_404_NOT_FOUND, code, message)


def payload_too_large(message: str) -> AppError:
    code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
    return AppError(code, "FILE_TOO_LARGE", message)
