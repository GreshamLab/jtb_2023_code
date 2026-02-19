import inferelator_velocity as ifv

from jtb_2023_code.figure_constants import (
    CC_TIME_COL,
    CC_LENGTH,
    RAPA_TIME_COL
)


def calculate_velocities(
    data,
    layer="X",
    graph="noise2self_distance_graph",
    force=False,
    wrap_cc_time=CC_LENGTH
):
    groups = [
        ("cell_cycle", CC_TIME_COL),
        ("rapamycin", RAPA_TIME_COL)
    ]

    for velo_key, time_key in groups:
        layer_out = velo_key + "_velocity"
        _wrap_time = None if time_key != "cell_cycle" else wrap_cc_time

        if layer_out not in data.layers or force:
            lref = data.X if layer == "X" else data.layers[layer]

            try:
                lref = lref.toarray()
            except AttributeError:
                pass

            print(f"Calculating velocities for {velo_key}:")

            data.layers[layer_out] = ifv.calc_velocity(
                lref,
                data.obs[time_key].values,
                data.obsp[graph],
                wrap_time=_wrap_time
            )

    return data
