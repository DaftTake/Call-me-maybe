"""Custom exceptions for the callme project."""




class CallMeMaybeError(Exception):
    """Base class for expected application errors."""


class UnknownFunctionError(CallMeMaybeError):
    """Raised when the model selects an unknown function."""


class DecodingError(CallMeMaybeError):
    """Raised when constrained decoding cannot continue safely."""
