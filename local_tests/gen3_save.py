"""Validate a Pokémon Emerald flash save the way the game itself does.

Format and checksum follow pret/pokeemerald src/save.c + include/save.h:
sectors are 0x1000 bytes (3968 data + 128 footer); footer holds id (u16) at
0xFF4, checksum (u16) at 0xFF6, signature 0x08012025 (u32) at 0xFF8 and save
counter (u32) at 0xFFC. Slot 1 = sectors 0-13, slot 2 = sectors 14-27. The game
zeroes each sector before copying data in, so summing all 3968 data bytes gives
the same checksum as summing only the struct's own size.

Usage: python gen3_save.py <file.sav> [...]   (exit 1 if no slot is valid)
"""
import struct
import sys

SECTOR_SIZE = 0x1000
SECTOR_DATA_SIZE = 3968
SECTORS_PER_SLOT = 14
SIGNATURE = 0x08012025


def checksum(data: bytes) -> int:
    total = sum(struct.unpack_from(f"<{SECTOR_DATA_SIZE // 4}I", data)) & 0xFFFFFFFF
    return ((total >> 16) + total) & 0xFFFF


def slot_status(save: bytes, slot: int) -> dict:
    """Return {'valid', 'counter', 'problems'} for save slot 0 or 1."""
    problems, ids, counters = [], set(), set()
    for n in range(slot * SECTORS_PER_SLOT, (slot + 1) * SECTORS_PER_SLOT):
        sector = save[n * SECTOR_SIZE:(n + 1) * SECTOR_SIZE]
        if len(sector) < SECTOR_SIZE:
            problems.append(f"sector {n}: truncated file")
            continue
        sid, csum, sig, counter = struct.unpack_from("<HHII", sector, 0xFF4)
        if sig != SIGNATURE:
            problems.append(f"sector {n}: no signature")
            continue
        if sid >= SECTORS_PER_SLOT:
            problems.append(f"sector {n}: bad id {sid}")
            continue
        if checksum(sector) != csum:
            problems.append(f"sector {n} (id {sid}): checksum mismatch")
        ids.add(sid)
        counters.add(counter)
    if len(ids) != SECTORS_PER_SLOT and not any("no signature" in p for p in problems):
        problems.append(f"missing section ids {sorted(set(range(SECTORS_PER_SLOT)) - ids)}")
    if len(counters) > 1:
        problems.append(f"mixed save counters {sorted(counters)}")
    valid = not problems
    return {"valid": valid, "counter": max(counters) if valid else None, "problems": problems}


def report(save: bytes) -> dict:
    slots = [slot_status(save, 0), slot_status(save, 1)]
    good = [s["counter"] for s in slots if s["valid"]]
    return {"size": len(save), "slots": slots, "newest_valid": max(good) if good else None}


if __name__ == "__main__":
    ok = True
    for path in sys.argv[1:]:
        r = report(open(path, "rb").read())
        print(f"{path}: {r['size']} bytes, newest valid save #{r['newest_valid']}")
        for i, s in enumerate(r["slots"], 1):
            detail = "OK" if s["valid"] else "; ".join(s["problems"][:4])
            print(f"  slot {i}: {'save #' + str(s['counter']) if s['valid'] else 'INVALID'} - {detail}")
        ok &= r["newest_valid"] is not None
    sys.exit(0 if ok else 1)
