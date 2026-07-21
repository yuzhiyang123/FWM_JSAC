__all__ = [
    "build_independent_channel_dataloaders",
    "build_independent_channel_three_way_dataloaders",
    "build_time_band_channel_dataloaders",
    "build_time_band_channel_three_way_dataloaders",
    "build_bs_slicing_data_dataloaders",
    "build_bs_slicing_data_partitions",
    "build_locationing_data_dataloaders",
    "build_locationing_data_partitions",
]


def __getattr__(name):
    if name == "build_time_band_channel_dataloaders":
        from .get_sequential_data import build_time_band_channel_dataloaders
        return build_time_band_channel_dataloaders
    if name == "build_time_band_channel_three_way_dataloaders":
        from .get_sequential_data import build_time_band_channel_three_way_dataloaders
        return build_time_band_channel_three_way_dataloaders
    if name == "build_independent_channel_dataloaders":
        from .get_simple_data import build_independent_channel_dataloaders
        return build_independent_channel_dataloaders
    if name == "build_independent_channel_three_way_dataloaders":
        from .get_simple_data import build_independent_channel_three_way_dataloaders
        return build_independent_channel_three_way_dataloaders
    if name == "build_bs_slicing_data_dataloaders":
        from .get_BS_slicing_data import build_bs_slicing_data_dataloaders
        return build_bs_slicing_data_dataloaders
    if name == "build_bs_slicing_data_partitions":
        from .get_BS_slicing_data import build_bs_slicing_data_partitions
        return build_bs_slicing_data_partitions
    if name == "build_locationing_data_dataloaders":
        from .get_locationing_data import build_locationing_data_dataloaders
        return build_locationing_data_dataloaders
    if name == "build_locationing_data_partitions":
        from .get_locationing_data import build_locationing_data_partitions
        return build_locationing_data_partitions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
