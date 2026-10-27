from egomimic.pl_utils.pl_data_utils_v5 import (
    EpisodeUniformWindowSampler,
    FullHorizonSequentialSampler,
)


class _FakeEpisodeBalancedDataset:
    _global_indices_by_dataset = {
        "short": list(range(0, 150)),
        "long": list(range(150, 750)),
    }


def test_episode_sampler_does_not_oversample_long_episode():
    sampler = EpisodeUniformWindowSampler(
        _FakeEpisodeBalancedDataset(),
        samples_per_episode=6,
        future_horizon=30,
        min_query_spacing=10,
        seed=7,
    )
    values = list(iter(sampler))
    assert len(values) == 12
    assert sum(value < 150 for value in values) == 6
    assert sum(value >= 150 for value in values) == 6
    assert max(value for value in values if value < 150) <= 120
    assert max(value for value in values if value >= 150) <= 720
    refreshed = list(iter(sampler))
    assert len(refreshed) == 12
    assert sum(value < 150 for value in refreshed) == 6
    assert sum(value >= 150 for value in refreshed) == 6
    assert values != refreshed


def test_validation_sampler_excludes_padded_tail():
    sampler = FullHorizonSequentialSampler(
        _FakeEpisodeBalancedDataset(), future_horizon=30
    )
    values = list(iter(sampler))
    assert sum(value < 150 for value in values) == 121
    assert sum(value >= 150 for value in values) == 571
    assert 121 not in values
    assert 721 not in values
