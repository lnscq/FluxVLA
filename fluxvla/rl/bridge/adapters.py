"""Environment preprocessing factories, independent of model dispatch."""


def _libero(flux_cfg, options, norm_stats):
    from .observation import LiberoObservationAdapter

    return LiberoObservationAdapter(
        flux_cfg,
        norm_stats=norm_stats,
        tokenizer_path=options.get('tokenizer_path'))


def _robotwin(flux_cfg, options, norm_stats):
    from .robotwin_observation import RoboTwinObservationAdapter

    return RoboTwinObservationAdapter(
        norm_stats=norm_stats,
        tokenizer_path=options.tokenizer_path,
        image_size=options.get('image_size', 224),
        max_token_len=options.get('max_token_len', 200))


OBSERVATION_ADAPTERS = {'libero': _libero, 'robotwin': _robotwin}


def build_observation_adapter(flux_cfg, options, norm_stats=None):
    name = options.get('observation_adapter', 'libero')
    try:
        factory = OBSERVATION_ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f'Unknown observation adapter: {name}') from error
    return factory(flux_cfg, options, norm_stats)
