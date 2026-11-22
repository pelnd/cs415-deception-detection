"""Step 9: leave-one-game-out (LOGO) folds over train + val.

The test games (4 and 12) are never part of a fold, so they stay for final results.

    from folds import logo_folds
    for game, train_idx, test_idx in logo_folds(dev):   # dev = train + val
        ...
"""
import numpy as np


def logo_folds(df):
    """Yield (held-out game_id, train row positions, held-out row positions), one fold per game."""
    games = np.sort(df.game_id.unique())
    g = df.game_id.values
    for game in games:
        yield int(game), np.where(g != game)[0], np.where(g == game)[0]


def check_folds(df):
    """Sanity checks: every row is held out exactly once and no game is on both sides."""
    held = np.zeros(len(df), int)
    for game, tr, te in logo_folds(df):
        assert set(df.game_id.iloc[te]) == {game}
        assert game not in set(df.game_id.iloc[tr])
        held[te] += 1
    assert (held == 1).all()
    return True
