def space_model_soccer(space_model, *args, **kwargs):
    if space_model == "soccer_OBSO":
        from .obso.soccer_obso_main_class import soccer_obso
        return soccer_obso(*args, **kwargs)
    elif space_model == "soccer_BIMOS":
        from .bimos.soccer_bimos_main_class import soccer_bimos
        return soccer_bimos(*args, **kwargs)
    else:
        raise NotImplementedError("Other soccer models are not implemented yet")