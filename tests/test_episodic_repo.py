# tests/test_episodic_repo.py
from domain.models import Episode, EpisodeEvent, Provenance
from infra.episodic_repo import EpisodicRepo

def _ep(
    id_: str,
    participants: list[str],
    summary: str,
    events: list[EpisodeEvent],
) -> Episode:
    return Episode(
        id=id_,
        participants=participants,
        summary=summary,
        events=events,
        provenance=Provenance(source="test"),
        meta={},
    )

def test_save_and_search_by_user_and_entities():
    repo = EpisodicRepo()  # :memory:
    ep1 = _ep(
        "ep1",
        ["Alice", "Nikolai"],
        "Evening near the lighthouse",
        [
            EpisodeEvent(t=1.0, event_type="seen", summary="Alice talked to fisherman Nikolai", refs={}),
            EpisodeEvent(t=2.0, event_type="walk", summary="They walked towards the lighthouse", refs={}),
        ],
    )
    ep2 = _ep(
        "ep2",
        ["Bob"],
        "Morning on the bridge",
        [
            EpisodeEvent(t=1.0, event_type="cross", summary="Bob crossed the old bridge", refs={}),
        ],
    )
    repo.save_episode(ep1)
    repo.save_episode(ep2)

    # Ищем для Alice c ключевым словом lighthouse
    out = repo.search(user="Alice", entities=["lighthouse"], k=5)
    ids = [e.id for e in out]
    assert "ep1" in ids
    assert "ep2" not in ids

def test_upsert_episode_overwrites_events():
    repo = EpisodicRepo()
    ep = _ep(
        "epx",
        ["Alice"],
        "First version",
        [EpisodeEvent(t=1.0, event_type="note", summary="v1", refs={})],
    )
    repo.save_episode(ep)

    ep_updated = _ep(
        "epx",
        ["Alice"],
        "Second version",
        [EpisodeEvent(t=2.0, event_type="note", summary="v2", refs={})],
    )
    repo.save_episode(ep_updated)

    out = repo.search(user="Alice", entities=[], k=1)
    assert len(out) == 1
    assert out[0].summary == "Second version"
    assert len(out[0].events) == 1
    assert out[0].events[0].summary == "v2"
