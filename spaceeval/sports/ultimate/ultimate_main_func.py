def space_model_ultimate(space_model, *args, **kwargs):
    if space_model == "wUPPCF":
        from .wuppcf.ultimate_wuppcf_main_class import ultimate_wuppcf

        return ultimate_wuppcf(*args, **kwargs)
    else:
        raise NotImplementedError("Other ultimate models are not implemented yet")
