"""David's HM protocol: which party Pokémon carries which HM, and which carriers fight.

The one table for it, read by the server's step hook (maintain_party: a carrier that
isn't a fighter never leads outside training) and by memory_reader.pick_fighter (such a
carrier is sent in only when nobody else wins).

Seen live (2026-11-04): TROPIUS (Lv27, the Fly carrier) led into a Team Magma
trainer's TORKOAL and fainted. The DANGER line said to switch to AZUMARILL (SURF 2x,
wins in 1); the 8B model attacked anyway. Losing the Fly carrier strands the run.
"""

# HM move -> the species that carries it (David's protocol, 2026-11-04).
HM_CARRIERS: dict[str, str] = {
    "SURF": "AZUMARILL",
    "STRENGTH": "AZUMARILL",
    "WATERFALL": "AZUMARILL",
    "DIVE": "AZUMARILL",
    "FLY": "TROPIUS",
    "CUT": "TROPIUS",
    "FLASH": "TROPIUS",
}

# Carrier species -> whether it is a real fighter. A fighter (AZUMARILL) battles like
# anyone else; one that isn't (TROPIUS) is kept out of harm's way.
HM_CARRIER_FIGHTS: dict[str, bool] = {
    "AZUMARILL": True,
    "TROPIUS": False,
}


def sits_out(species: str) -> bool:
    """True for an HM carrier that is not a fighter: it shouldn't lead or be sent in."""
    return HM_CARRIER_FIGHTS.get(str(species).upper()) is False


def carried_hms(species: str) -> list:
    """The HMs this species carries, in table order ("FLY, CUT, FLASH")."""
    name = str(species).upper()
    return [move for move, carrier in HM_CARRIERS.items() if carrier == name]
