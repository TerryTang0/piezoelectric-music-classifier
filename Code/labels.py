import os


GENRE_MAP = {
    "edm": "Electronic",
    "modern_pop": "Modern Pop",
    "piano": "Piano",
    "country": "Country",
    "r&b": "R&B",
    "rap": "Rap",
}


def resolve_genre_and_song(filename):
    """Return the genre label and song ID from a filename."""

    clean_name = filename.lower()
    song_id = os.path.splitext(filename)[0].strip()

    # Baseline recordings are only used for calibration
    if clean_name.startswith("baseline_") or "no music" in clean_name:
        return "Baseline_Calibration", song_id

    for prefix, genre in GENRE_MAP.items():
        if clean_name.startswith(prefix + "_"):
            return genre, song_id

    return "Unknown", song_id