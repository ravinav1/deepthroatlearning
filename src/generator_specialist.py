"""Select one output generator without reducing the model's input context."""

from features import HOURS


def output_targets(columns, generators, target_generator=None):
    if target_generator is None:
        return columns, generators
    assert target_generator in generators, f"Unknown target generator: {target_generator}"
    start = generators.index(target_generator) * HOURS
    selected = columns[start:start + HOURS]
    assert selected == [f"result_committed_{target_generator}_t{hour}" for hour in range(HOURS)]
    return selected, [target_generator]
