def clamp(value, min_value, max_value):
    """
    Clamp value between min_value and max_value.

    Example
    --------
    >>> clamp(10, 0, 5)
    5
    >>> clamp(-3, 0, 5)
    0
    >>> clamp(3, 0, 5)
    3
    
    Parameters
    ----------
    value : float
        Value to be clamped.
    min_value : float
        Minimum allowable value.
    max_value : float
        Maximum allowable value.

    Returns
    -------
    float
        Clamped value.
    """
    return max(min_value, min(value, max_value))