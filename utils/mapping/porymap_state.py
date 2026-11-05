"""Porymap mapping and formatting helpers extracted from state_formatter."""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.mapping.flag_hide_constants import FLAG_HIDE_IDS

logger = logging.getLogger(__name__)


@dataclass
class PorymapResult:
    context_parts: List[str]
    json_map: Optional[Dict[str, Any]] = None


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any], run_id: str = "run1") -> None:
    # region agent log
    try:
        import time
        payload = {
            "sessionId": "b11f04",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/data3/tu8435/thesis-remote/pokeagent-speedrun/.cursor/debug-b11f04.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # endregion

def _get_pokeemerald_root() -> Optional[Path]:
    """Get the pokeemerald root directory path. Delegates to centralized resolver."""
    from pokemon_env.porymap_paths import get_porymap_root
    return get_porymap_root()


# ROM location name to Porymap map name mapping
ROM_TO_PORYMAP_MAP = {
    # Intro/Cutscene locations
    "MOVING_VAN": "InsideOfTruck",  # Intro cutscene (moving van)
    
    # Towns
    "LITTLEROOT TOWN": "LittlerootTown",
    "OLDALE TOWN": "OldaleTown",
    "DEWFORD TOWN": "DewfordTown",
    "LAVARIDGE TOWN": "LavaridgeTown",
    "FALLARBOR TOWN": "FallarborTown",
    "VERDANTURF TOWN": "VerdanturfTown",
    "PACIFIDLOG TOWN": "PacifidlogTown",
    
    # Cities
    "PETALBURG CITY": "PetalburgCity",
    "SLATEPORT CITY": "SlateportCity",
    "MAUVILLE CITY": "MauvilleCity",
    "RUSTBORO CITY": "RustboroCity",
    "FORTREE CITY": "FortreeCity",
    "LILYCOVE CITY": "LilycoveCity",
    "MOSSDEEP CITY": "MossdeepCity",
    "SOOTOPOLIS CITY": "SootopolisCity",
    "EVER GRANDE CITY": "EverGrandeCity",
    
    # Routes
    "ROUTE 101": "Route101",
    "ROUTE 102": "Route102",
    "ROUTE 103": "Route103",
    "ROUTE 104": "Route104",
    "ROUTE 105": "Route105",
    "ROUTE 106": "Route106",
    "ROUTE 107": "Route107",
    "ROUTE 108": "Route108",
    "ROUTE 109": "Route109",
    "ROUTE 110": "Route110",
    "ROUTE 111": "Route111",
    "ROUTE 112": "Route112",
    "ROUTE 113": "Route113",
    "ROUTE 114": "Route114",
    "ROUTE 115": "Route115",
    "ROUTE 116": "Route116",
    "ROUTE 117": "Route117",
    "ROUTE 118": "Route118",
    "ROUTE 119": "Route119",
    "ROUTE 120": "Route120",
    "ROUTE 121": "Route121",
    "ROUTE 122": "Route122",
    "ROUTE 123": "Route123",
    "ROUTE 124": "Route124",
    "ROUTE 125": "Route125",
    "ROUTE 126": "Route126",
    "ROUTE 127": "Route127",
    "ROUTE 128": "Route128",
    "ROUTE 129": "Route129",
    "ROUTE 130": "Route130",
    "ROUTE 131": "Route131",
    "ROUTE 132": "Route132",
    "ROUTE 133": "Route133",
    "ROUTE 134": "Route134",
    
    # Buildings (common patterns)
    "PETALBURG WOODS": "PetalburgWoods",
    "RUSTURF TUNNEL": "RusturfTunnel",

    # Professor Birch's Lab
    "LITTLEROOT TOWN PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
    "PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",

        # The folloing mappings were generated via cursor and verified
    "ABANDONED SHIP CAPTAINS OFFICE": "AbandonedShip_CaptainsOffice",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS 1F": "AbandonedShip_Corridors_1F",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS B1F": "AbandonedShip_Corridors_B1F",  # 100.0% match, verified
    "ABANDONED SHIP DECK": "AbandonedShip_Deck",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR CORRIDORS": "AbandonedShip_HiddenFloorCorridors",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR ROOMS": "AbandonedShip_HiddenFloorRooms",  # 100.0% match, verified
    "ABANDONED SHIP ROOM B1F": "AbandonedShip_Room_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS 1F": "AbandonedShip_Rooms_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS B1F": "AbandonedShip_Rooms_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 1F": "AbandonedShip_Rooms2_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 B1F": "AbandonedShip_Rooms2_B1F",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER1": "AbandonedShip_Underwater1",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER2": "AbandonedShip_Underwater2",  # 100.0% match, verified
    "ALTERING CAVE": "AlteringCave",  # 100.0% match, verified
    "ANCIENT TOMB": "AncientTomb",  # 100.0% match, verified
    "AQUA HIDEOUT 1F": "AquaHideout_1F",  # 100.0% match, verified
    "AQUA HIDEOUT B1F": "AquaHideout_B1F",  # 100.0% match, verified
    "AQUA HIDEOUT B2F": "AquaHideout_B2F",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP1": "AquaHideout_UnusedRubyMap1",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP2": "AquaHideout_UnusedRubyMap2",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP3": "AquaHideout_UnusedRubyMap3",  # 100.0% match, verified
    "ARTISAN CAVE 1F": "ArtisanCave_1F",  # 100.0% match, verified
    "ARTISAN CAVE B1F": "ArtisanCave_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN 1F": "CaveOfOrigin_1F",  # 100.0% match, verified
    "CAVE OF ORIGIN B1F": "CaveOfOrigin_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN ENTRANCE": "CaveOfOrigin_Entrance",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP1": "CaveOfOrigin_UnusedRubySapphireMap1",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP2": "CaveOfOrigin_UnusedRubySapphireMap2",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP3": "CaveOfOrigin_UnusedRubySapphireMap3",  # 100.0% match, verified
    "DESERT RUINS": "DesertRuins",  # 100.0% match, verified
    "DESERT UNDERPASS": "DesertUnderpass",  # 100.0% match, verified
    "DEWFORD TOWN GYM": "DewfordTown_Gym",  # 100.0% match, verified
    "DEWFORD TOWN HALL": "DewfordTown_Hall",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE1": "DewfordTown_House1",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE2": "DewfordTown_House2",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 1F": "DewfordTown_PokemonCenter_1F",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 2F": "DewfordTown_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY CHAMPIONS ROOM": "EverGrandeCity_ChampionsRoom",  # 100.0% match, verified
    "EVER GRANDE CITY DRAKES ROOM": "EverGrandeCity_DrakesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY GLACIAS ROOM": "EverGrandeCity_GlaciasRoom",  # 100.0% match, verified
    "EVER GRANDE CITY HALL OF FAME": "EverGrandeCity_HallOfFame",  # 100.0% match, verified
    "EVER GRANDE CITY HALL1": "EverGrandeCity_Hall1",  # 100.0% match, verified
    "EVER GRANDE CITY HALL2": "EverGrandeCity_Hall2",  # 100.0% match, verified
    "EVER GRANDE CITY HALL3": "EverGrandeCity_Hall3",  # 100.0% match, verified
    "EVER GRANDE CITY HALL4": "EverGrandeCity_Hall4",  # 100.0% match, verified
    "EVER GRANDE CITY HALL5": "EverGrandeCity_Hall5",  # 100.0% match, verified
    "EVER GRANDE CITY PHOEBES ROOM": "EverGrandeCity_PhoebesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 1F": "EverGrandeCity_PokemonCenter_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 2F": "EverGrandeCity_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 1F": "EverGrandeCity_PokemonLeague_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 2F": "EverGrandeCity_PokemonLeague_2F",  # 100.0% match, verified
    "EVER GRANDE CITY SIDNEYS ROOM": "EverGrandeCity_SidneysRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT BATTLE ROOM": "FallarborTown_BattleTentBattleRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT CORRIDOR": "FallarborTown_BattleTentCorridor",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT LOBBY": "FallarborTown_BattleTentLobby",  # 100.0% match, verified
    "FALLARBOR TOWN COZMOS HOUSE": "FallarborTown_CozmosHouse",  # 100.0% match, verified
    "FALLARBOR TOWN MART": "FallarborTown_Mart",  # 100.0% match, verified
    "FALLARBOR TOWN MOVE RELEARNERS HOUSE": "FallarborTown_MoveRelearnersHouse",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 1F": "FallarborTown_PokemonCenter_1F",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 2F": "FallarborTown_PokemonCenter_2F",  # 100.0% match, verified
    "FIERY PATH": "FieryPath",  # 100.0% match, verified
    "FIERY PATH INTERIOR": "FieryPath",  # Same location, different name variant
    "FORTREE CITY DECORATION SHOP": "FortreeCity_DecorationShop",  # 100.0% match, verified
    "FORTREE CITY GYM": "FortreeCity_Gym",  # 100.0% match, verified
    "FORTREE CITY HOUSE1": "FortreeCity_House1",  # 100.0% match, verified
    "FORTREE CITY HOUSE2": "FortreeCity_House2",  # 100.0% match, verified
    "FORTREE CITY HOUSE3": "FortreeCity_House3",  # 100.0% match, verified
    "FORTREE CITY HOUSE4": "FortreeCity_House4",  # 100.0% match, verified
    "FORTREE CITY HOUSE5": "FortreeCity_House5",  # 100.0% match, verified
    "FORTREE CITY MART": "FortreeCity_Mart",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 1F": "FortreeCity_PokemonCenter_1F",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 2F": "FortreeCity_PokemonCenter_2F",  # 100.0% match, verified
    "GRANITE CAVE 1F": "GraniteCave_1F",  # 100.0% match, verified
    "GRANITE CAVE B1F": "GraniteCave_B1F",  # 100.0% match, verified
    "GRANITE CAVE B2F": "GraniteCave_B2F",  # 100.0% match, verified
    "GRANITE CAVE STEVENS ROOM": "GraniteCave_StevensRoom",  # 100.0% match, verified
    "ISLAND CAVE": "IslandCave",  # 100.0% match, verified
    "JAGGED PASS": "JaggedPass",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM 1F": "LavaridgeTown_Gym_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM B1F": "LavaridgeTown_Gym_B1F",  # 100.0% match, verified
    "LAVARIDGE TOWN HERB SHOP": "LavaridgeTown_HerbShop",  # 100.0% match, verified
    "LAVARIDGE TOWN HOUSE": "LavaridgeTown_House",  # 100.0% match, verified
    "LAVARIDGE TOWN MART": "LavaridgeTown_Mart",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 1F": "LavaridgeTown_PokemonCenter_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 2F": "LavaridgeTown_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST HALL": "LilycoveCity_ContestHall",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST LOBBY": "LilycoveCity_ContestLobby",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 1F": "LilycoveCity_CoveLilyMotel_1F",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 2F": "LilycoveCity_CoveLilyMotel_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 1F": "LilycoveCity_DepartmentStore_1F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 2F": "LilycoveCity_DepartmentStore_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 3F": "LilycoveCity_DepartmentStore_3F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 4F": "LilycoveCity_DepartmentStore_4F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 5F": "LilycoveCity_DepartmentStore_5F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ELEVATOR": "LilycoveCity_DepartmentStoreElevator",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ROOFTOP": "LilycoveCity_DepartmentStoreRooftop",  # 100.0% match, verified
    "LILYCOVE CITY HARBOR": "LilycoveCity_Harbor",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE1": "LilycoveCity_House1",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE2": "LilycoveCity_House2",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE3": "LilycoveCity_House3",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE4": "LilycoveCity_House4",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 1F": "LilycoveCity_LilycoveMuseum_1F",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 2F": "LilycoveCity_LilycoveMuseum_2F",  # 100.0% match, verified
    "LILYCOVE CITY MOVE DELETERS HOUSE": "LilycoveCity_MoveDeletersHouse",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 1F": "LilycoveCity_PokemonCenter_1F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 2F": "LilycoveCity_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON TRAINER FAN CLUB": "LilycoveCity_PokemonTrainerFanClub",  # 100.0% match, verified
    "LILYCOVE CITY UNUSED MART": "LilycoveCity_UnusedMart",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 1F": "LittlerootTown_BrendansHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 2F": "LittlerootTown_BrendansHouse_2F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 1F": "LittlerootTown_MaysHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 2F": "LittlerootTown_MaysHouse_2F",  # 100.0% match, verified
    "MAGMA HIDEOUT 1F": "MagmaHideout_1F",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 1R": "MagmaHideout_2F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 2R": "MagmaHideout_2F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 3R": "MagmaHideout_2F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 1R": "MagmaHideout_3F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 2R": "MagmaHideout_3F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 3R": "MagmaHideout_3F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 4F": "MagmaHideout_4F",  # 100.0% match, verified
    "MARINE CAVE END": "MarineCave_End",  # 100.0% match, verified
    "MARINE CAVE ENTRANCE": "MarineCave_Entrance",  # 100.0% match, verified
    "MAUVILLE CITY BIKE SHOP": "MauvilleCity_BikeShop",  # 100.0% match, verified
    "MAUVILLE CITY GAME CORNER": "MauvilleCity_GameCorner",  # 100.0% match, verified
    "MAUVILLE CITY GYM": "MauvilleCity_Gym",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE1": "MauvilleCity_House1",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE2": "MauvilleCity_House2",  # 100.0% match, verified
    "MAUVILLE CITY MART": "MauvilleCity_Mart",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 1F": "MauvilleCity_PokemonCenter_1F",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 2F": "MauvilleCity_PokemonCenter_2F",  # 100.0% match, verified
    "METEOR FALLS 1F 1R": "MeteorFalls_1F_1R",  # 100.0% match, verified
    "METEOR FALLS 1F 2R": "MeteorFalls_1F_2R",  # 100.0% match, verified
    "METEOR FALLS B1F 1R": "MeteorFalls_B1F_1R",  # 100.0% match, verified
    "METEOR FALLS B1F 2R": "MeteorFalls_B1F_2R",  # 100.0% match, verified
    "METEOR FALLS STEVENS CAVE": "MeteorFalls_StevensCave",  # 100.0% match, verified
    "MIRAGE TOWER 1F": "MirageTower_1F",  # 100.0% match, verified
    "MIRAGE TOWER 2F": "MirageTower_2F",  # 100.0% match, verified
    "MIRAGE TOWER 3F": "MirageTower_3F",  # 100.0% match, verified
    "MIRAGE TOWER 4F": "MirageTower_4F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER 1F": "MossdeepCity_GameCorner_1F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER B1F": "MossdeepCity_GameCorner_B1F",  # 100.0% match, verified
    "MOSSDEEP CITY GYM": "MossdeepCity_Gym",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE1": "MossdeepCity_House1",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE2": "MossdeepCity_House2",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE3": "MossdeepCity_House3",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE4": "MossdeepCity_House4",  # 100.0% match, verified
    "MOSSDEEP CITY MART": "MossdeepCity_Mart",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 1F": "MossdeepCity_PokemonCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 2F": "MossdeepCity_PokemonCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 1F": "MossdeepCity_SpaceCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 2F": "MossdeepCity_SpaceCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY STEVENS HOUSE": "MossdeepCity_StevensHouse",  # 100.0% match, verified
    "MT CHIMNEY": "MtChimney",  # 100.0% match, verified
    "MT CHIMNEY CABLE CAR STATION": "MtChimney_CableCarStation",  # 100.0% match, verified
    "MT PYRE 1F": "MtPyre_1F",  # 100.0% match, verified
    "MT PYRE 2F": "MtPyre_2F",  # 100.0% match, verified
    "MT PYRE 3F": "MtPyre_3F",  # 100.0% match, verified
    "MT PYRE 4F": "MtPyre_4F",  # 100.0% match, verified
    "MT PYRE 5F": "MtPyre_5F",  # 100.0% match, verified
    "MT PYRE 6F": "MtPyre_6F",  # 100.0% match, verified
    "MT PYRE EXTERIOR": "MtPyre_Exterior",  # 100.0% match, verified
    "MT PYRE SUMMIT": "MtPyre_Summit",  # 100.0% match, verified
    "NEW MAUVILLE ENTRANCE": "NewMauville_Entrance",  # 100.0% match, verified
    "NEW MAUVILLE INSIDE": "NewMauville_Inside",  # 100.0% match, verified
    "OLDALE TOWN HOUSE1": "OldaleTown_House1",  # 100.0% match, verified
    "OLDALE TOWN HOUSE2": "OldaleTown_House2",  # 100.0% match, verified
    "OLDALE TOWN MART": "OldaleTown_Mart",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 1F": "OldaleTown_PokemonCenter_1F",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 2F": "OldaleTown_PokemonCenter_2F",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE1": "PacifidlogTown_House1",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE2": "PacifidlogTown_House2",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE3": "PacifidlogTown_House3",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE4": "PacifidlogTown_House4",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE5": "PacifidlogTown_House5",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 1F": "PacifidlogTown_PokemonCenter_1F",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 2F": "PacifidlogTown_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY GYM": "PetalburgCity_Gym",  # 100.0% match, verified
    "PETALBURG CITY HOUSE1": "PetalburgCity_House1",  # 100.0% match, verified
    "PETALBURG CITY HOUSE2": "PetalburgCity_House2",  # 100.0% match, verified
    "PETALBURG CITY MART": "PetalburgCity_Mart",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 1F": "PetalburgCity_PokemonCenter_1F",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 2F": "PetalburgCity_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY WALLYS HOUSE": "PetalburgCity_WallysHouse",  # 100.0% match, verified
    "ROUTE 104 MR BRINEYS HOUSE": "Route104_MrBrineysHouse",  # 100.0% match, verified
    "ROUTE 104 PRETTY PETAL FLOWER SHOP": "Route104_PrettyPetalFlowerShop",  # 100.0% match, verified
    "ROUTE 109 SEASHORE HOUSE": "Route109_SeashoreHouse",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE CORRIDOR": "Route110_TrickHouseCorridor",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE END": "Route110_TrickHouseEnd",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE ENTRANCE": "Route110_TrickHouseEntrance",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE1": "Route110_TrickHousePuzzle1",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE2": "Route110_TrickHousePuzzle2",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE3": "Route110_TrickHousePuzzle3",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE4": "Route110_TrickHousePuzzle4",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE5": "Route110_TrickHousePuzzle5",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE6": "Route110_TrickHousePuzzle6",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE7": "Route110_TrickHousePuzzle7",  # 100.0% match, verified
    "ROUTE 111 OLD LADYS REST STOP": "Route111_OldLadysRestStop",  # 100.0% match, verified
    "ROUTE 111 WINSTRATE FAMILYS HOUSE": "Route111_WinstrateFamilysHouse",  # 100.0% match, verified
    "ROUTE 112 CABLE CAR STATION": "Route112_CableCarStation",  # 100.0% match, verified
    "ROUTE 113 GLASS WORKSHOP": "Route113_GlassWorkshop",  # 100.0% match, verified
    "ROUTE 114 FOSSIL MANIACS HOUSE": "Route114_FossilManiacsHouse",  # 100.0% match, verified
    "ROUTE 114 FOSSIL MANIACS TUNNEL": "Route114_FossilManiacsTunnel",  # 100.0% match, verified
    "ROUTE 114 LANETTES HOUSE": "Route114_LanettesHouse",  # 100.0% match, verified
    "ROUTE 116 TUNNELERS REST HOUSE": "Route116_TunnelersRestHouse",  # 100.0% match, verified
    "ROUTE 117 POKEMON DAY CARE": "Route117_PokemonDayCare",  # 100.0% match, verified
    "ROUTE 119 HOUSE": "Route119_House",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 1F": "Route119_WeatherInstitute_1F",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 2F": "Route119_WeatherInstitute_2F",  # 100.0% match, verified
    "ROUTE 121 SAFARI ZONE ENTRANCE": "Route121_SafariZoneEntrance",  # 100.0% match, verified
    "ROUTE 123 BERRY MASTERS HOUSE": "Route123_BerryMastersHouse",  # 100.0% match, verified
    "ROUTE 124 DIVING TREASURE HUNTERS HOUSE": "Route124_DivingTreasureHuntersHouse",  # 100.0% match, verified
    "RUSTBORO CITY CUTTERS HOUSE": "RustboroCity_CuttersHouse",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 1F": "RustboroCity_DevonCorp_1F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 2F": "RustboroCity_DevonCorp_2F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 3F": "RustboroCity_DevonCorp_3F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 1F": "RustboroCity_Flat1_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 2F": "RustboroCity_Flat1_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 1F": "RustboroCity_Flat2_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 2F": "RustboroCity_Flat2_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 3F": "RustboroCity_Flat2_3F",  # 100.0% match, verified
    "RUSTBORO CITY GYM": "RustboroCity_Gym",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE1": "RustboroCity_House1",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE2": "RustboroCity_House2",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE3": "RustboroCity_House3",  # 100.0% match, verified
    "RUSTBORO CITY MART": "RustboroCity_Mart",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 1F": "RustboroCity_PokemonCenter_1F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 2F": "RustboroCity_PokemonCenter_2F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON SCHOOL": "RustboroCity_PokemonSchool",  # 100.0% match, verified
    "SCORCHED SLAB": "ScorchedSlab",  # 100.0% match, verified
    "SEAFLOOR CAVERN ENTRANCE": "SeafloorCavern_Entrance",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM1": "SeafloorCavern_Room1",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM2": "SeafloorCavern_Room2",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM3": "SeafloorCavern_Room3",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM4": "SeafloorCavern_Room4",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM5": "SeafloorCavern_Room5",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM6": "SeafloorCavern_Room6",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM7": "SeafloorCavern_Room7",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM8": "SeafloorCavern_Room8",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM9": "SeafloorCavern_Room9",  # 100.0% match, verified
    "SEALED CHAMBER INNER ROOM": "SealedChamber_InnerRoom",  # 100.0% match, verified
    "SEALED CHAMBER OUTER ROOM": "SealedChamber_OuterRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE ENTRANCE ROOM": "ShoalCave_HighTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE INNER ROOM": "ShoalCave_HighTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ENTRANCE ROOM": "ShoalCave_LowTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ICE ROOM": "ShoalCave_LowTideIceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE INNER ROOM": "ShoalCave_LowTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE LOWER ROOM": "ShoalCave_LowTideLowerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE STAIRS ROOM": "ShoalCave_LowTideStairsRoom",  # 100.0% match, verified
    "SKY PILLAR 1F": "SkyPillar_1F",  # 100.0% match, verified
    "SKY PILLAR 2F": "SkyPillar_2F",  # 100.0% match, verified
    "SKY PILLAR 3F": "SkyPillar_3F",  # 100.0% match, verified
    "SKY PILLAR 4F": "SkyPillar_4F",  # 100.0% match, verified
    "SKY PILLAR 5F": "SkyPillar_5F",  # 100.0% match, verified
    "SKY PILLAR ENTRANCE": "SkyPillar_Entrance",  # 100.0% match, verified
    "SKY PILLAR OUTSIDE": "SkyPillar_Outside",  # 100.0% match, verified
    "SKY PILLAR TOP": "SkyPillar_Top",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT BATTLE ROOM": "SlateportCity_BattleTentBattleRoom",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT CORRIDOR": "SlateportCity_BattleTentCorridor",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT LOBBY": "SlateportCity_BattleTentLobby",  # 100.0% match, verified
    "SLATEPORT CITY HARBOR": "SlateportCity_Harbor",  # 100.0% match, verified
    "SLATEPORT CITY HOUSE": "SlateportCity_House",  # 100.0% match, verified
    "SLATEPORT CITY MART": "SlateportCity_Mart",  # 100.0% match, verified
    "SLATEPORT CITY NAME RATERS HOUSE": "SlateportCity_NameRatersHouse",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 1F": "SlateportCity_OceanicMuseum_1F",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 2F": "SlateportCity_OceanicMuseum_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 1F": "SlateportCity_PokemonCenter_1F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 2F": "SlateportCity_PokemonCenter_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON FAN CLUB": "SlateportCity_PokemonFanClub",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 1F": "SlateportCity_SternsShipyard_1F",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 2F": "SlateportCity_SternsShipyard_2F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM 1F": "SootopolisCity_Gym_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM B1F": "SootopolisCity_Gym_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE1": "SootopolisCity_House1",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE2": "SootopolisCity_House2",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE3": "SootopolisCity_House3",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE4": "SootopolisCity_House4",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE5": "SootopolisCity_House5",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE6": "SootopolisCity_House6",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE7": "SootopolisCity_House7",  # 100.0% match, verified
    "SOOTOPOLIS CITY LOTAD AND SEEDOT HOUSE": "SootopolisCity_LotadAndSeedotHouse",  # 100.0% match, verified
    "SOOTOPOLIS CITY MART": "SootopolisCity_Mart",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE 1F": "SootopolisCity_MysteryEventsHouse_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE B1F": "SootopolisCity_MysteryEventsHouse_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 1F": "SootopolisCity_PokemonCenter_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 2F": "SootopolisCity_PokemonCenter_2F",  # 100.0% match, verified
    "TERRA CAVE END": "TerraCave_End",  # 100.0% match, verified
    "TERRA CAVE ENTRANCE": "TerraCave_Entrance",  # 100.0% match, verified
    "UNDERWATER MARINE CAVE": "Underwater_MarineCave",  # 100.0% match, verified
    "UNDERWATER ROUTE 105": "Underwater_Route105",  # 100.0% match, verified
    "UNDERWATER ROUTE 124": "Underwater_Route124",  # 100.0% match, verified
    "UNDERWATER ROUTE 125": "Underwater_Route125",  # 100.0% match, verified
    "UNDERWATER ROUTE 126": "Underwater_Route126",  # 100.0% match, verified
    "UNDERWATER ROUTE 127": "Underwater_Route127",  # 100.0% match, verified
    "UNDERWATER ROUTE 128": "Underwater_Route128",  # 100.0% match, verified
    "UNDERWATER ROUTE 129": "Underwater_Route129",  # 100.0% match, verified
    "UNDERWATER ROUTE 134": "Underwater_Route134",  # 100.0% match, verified
    "UNDERWATER SEAFLOOR CAVERN": "Underwater_SeafloorCavern",  # 100.0% match, verified
    "UNDERWATER SEALED CHAMBER": "Underwater_SealedChamber",  # 100.0% match, verified
    "UNDERWATER SOOTOPOLIS CITY": "Underwater_SootopolisCity",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT BATTLE ROOM": "VerdanturfTown_BattleTentBattleRoom",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT CORRIDOR": "VerdanturfTown_BattleTentCorridor",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT LOBBY": "VerdanturfTown_BattleTentLobby",  # 100.0% match, verified
    "VERDANTURF TOWN FRIENDSHIP RATERS HOUSE": "VerdanturfTown_FriendshipRatersHouse",  # 100.0% match, verified
    "VERDANTURF TOWN HOUSE": "VerdanturfTown_House",  # 100.0% match, verified
    "VERDANTURF TOWN MART": "VerdanturfTown_Mart",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 1F": "VerdanturfTown_PokemonCenter_1F",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 2F": "VerdanturfTown_PokemonCenter_2F",  # 100.0% match, verified
    "VERDANTURF TOWN WANDAS HOUSE": "VerdanturfTown_WandasHouse",  # 100.0% match, verified
    "VICTORY ROAD 1F": "VictoryRoad_1F",  # 100.0% match, verified
    "VICTORY ROAD B1F": "VictoryRoad_B1F",  # 100.0% match, verified
    "VICTORY ROAD B2F": "VictoryRoad_B2F",  # 100.0% match, verified
}

def _get_porymap_map_name(location_name: Optional[str]) -> Optional[str]:
    """Convert ROM location name to porymap map name."""
    if not location_name:
        return None
    return ROM_TO_PORYMAP_MAP.get(location_name)


def _normalize_local_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _get_runtime_xy(runtime_obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    x = runtime_obj.get("current_x", runtime_obj.get("x"))
    y = runtime_obj.get("current_y", runtime_obj.get("y"))
    return x, y


def _graphics_ids_compatible(static_gid: Any, runtime_gid: Any) -> bool:
    """Return True when graphics IDs are comparable and match, or unknown."""
    if static_gid is None or runtime_gid is None:
        return True
    if isinstance(static_gid, str) and isinstance(runtime_gid, str):
        return static_gid == runtime_gid
    # Runtime IDs are often numeric while porymap IDs are symbolic strings.
    # Treat cross-type as unknown instead of rejecting a potential match.
    return True


NON_NPC_GRAPHICS_KEYWORDS = {
    "TRUCK",
}


def _is_item_object_event(obj: Dict[str, Any]) -> bool:
    """Best-effort item object detection."""
    graphics_id = str(obj.get("graphics_id", "")).upper()
    if "ITEM_BALL" in graphics_id or "ITEMBALL" in graphics_id:
        return True

    script_name = str(obj.get("script", "")).upper()
    if "ITEMBALL" in script_name or "ITEM_BALL" in script_name:
        return True
    return False


def _is_npc_object_event(obj: Dict[str, Any]) -> bool:
    """Best-effort NPC detection without assuming unknowns are items."""
    if _is_item_object_event(obj):
        return False

    graphics_id = str(obj.get("graphics_id", "")).upper()
    if any(keyword in graphics_id for keyword in NON_NPC_GRAPHICS_KEYWORDS):
        return False

    trainer_type = str(obj.get("trainer_type", "")).upper()
    if trainer_type and trainer_type != "TRAINER_TYPE_NONE":
        return True

    movement_type = str(obj.get("movement_type", "")).upper()
    if movement_type and movement_type != "MOVEMENT_TYPE_NONE":
        return True

    local_id = str(obj.get("local_id", "")).strip()
    if local_id:
        return True

    # Most character object events use OBJ_EVENT_GFX_*.
    if graphics_id.startswith("OBJ_EVENT_GFX_"):
        return True
    return False


BOULDER_GRAPHICS = "OBJ_EVENT_GFX_PUSHABLE_BOULDER"
# Statues are object events with no script (pret MossdeepCity_Gym map.json: 21 of them,
# script 0x0): they block the way like a person, and A in front of one does nothing.
STATUE_GRAPHICS = frozenset({"OBJ_EVENT_GFX_TRICK_HOUSE_STATUE"})
STATUE_MARKER = "Z"


def objects_where_they_stand(
    objects: List[Dict[str, Any]], places: Dict[int, Tuple[int, int]], offset: Optional[Tuple[int, int]] = None
) -> List[Dict[str, Any]]:
    """The objects with x, y set to where they stand now: places is {local id: tile}
    (server.field_control.object_places), matched by each object's "event_id"; offset
    turns map tiles into the walking tools' (local = map - offset). Objects without an
    event_id (an override map's own) and Strength boulders (their pushes are read apart,
    _boulder_places) keep their tile."""
    ox, oy = offset or (0, 0)
    out = []
    for obj in objects:
        place = places.get(obj.get("event_id")) if obj.get("graphics_id") != BOULDER_GRAPHICS else None
        out.append({**obj, "x": place[0] - ox, "y": place[1] - oy} if place is not None else obj)
    return out


def _object_ascii_marker(obj: Dict[str, Any]) -> str:
    """Choose ASCII marker for reconciled object events."""
    if obj.get("graphics_id") == BOULDER_GRAPHICS:
        return "R"  # a Strength boulder, not a person to talk to
    if obj.get("graphics_id") in STATUE_GRAPHICS:
        return STATUE_MARKER  # in the way like a person, but nobody to talk to
    if _is_item_object_event(obj):
        return "I"
    if _is_npc_object_event(obj):
        return "N"
    return "O"


# Tiles a player can't stand on when looking for openings on a map edge.
_EDGE_BLOCKED = {"#", "X", "W"}


def _readable_map_name(map_const: str) -> str:
    """MAP_LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB -> LITTLEROOT TOWN PROFESSOR BIRCHS LAB.

    The same name memory_reader.read_location gives that map (MapLocation's member name
    with spaces), so an exit's name is the location the player reads after taking it.
    """
    name = str(map_const)
    if name.startswith("MAP_"):
        name = name[len("MAP_"):]
    # pokeemerald writes ROUTE101; the rest of the prompt says ROUTE 101. Only routes:
    # spacing every letter-digit pair made GRANITE_CAVE_B1F "GRANITE CAVE B 1F", an exit
    # name that never equalled the GRANITE CAVE B1F the player then stood on.
    out = []
    i = 0
    while i < len(name):
        if name.startswith("ROUTE", i) and name[i + 5:i + 6].isdigit():
            out.append("ROUTE ")
            i += len("ROUTE")
        else:
            out.append(name[i])
            i += 1
    return "".join(out).replace("_", " ")


@lru_cache(maxsize=1)
def _map_dirs_by_const() -> Dict[str, str]:
    """MAP_ROUTE101 -> Route101 (the data/maps directory), from each map.json's id."""
    root = _get_pokeemerald_root()
    out: Dict[str, str] = {}
    if not root:
        return out
    for map_file in (root / "data" / "maps").glob("*/map.json"):
        try:
            with open(map_file, encoding="utf-8") as fh:
                map_id = json.load(fh).get("id")
        except (OSError, ValueError):
            continue
        if map_id:
            out[map_id] = map_file.parent.name
    return out


@lru_cache(maxsize=1)
def _map_group_dirs() -> Tuple[Tuple[str, ...], ...]:
    """data/maps/map_groups.json as ((dir, ...) per group, ...): the ROM's map numbering."""
    root = _get_pokeemerald_root()
    if not root:
        return ()
    try:
        with open(root / "data" / "maps" / "map_groups.json", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ()
    return tuple(tuple(data.get(group, ())) for group in data.get("group_order", ()))


def decomp_map_at(group: int, num: int) -> Optional[Tuple[str, str]]:
    """(data/maps directory, readable name) of map (group, num) as the ROM numbers it.

    Straight from the decomp's map_groups.json, which MapLocation is generated from, so the
    readable name is the one read_location gives for the same (group, num).
    """
    groups = _map_group_dirs()
    if not (0 <= group < len(groups) and 0 <= num < len(groups[group])):
        return None
    map_dir = groups[group][num]
    const = next((c for c, d in _map_dirs_by_const().items() if d == map_dir), None)
    readable = _readable_map_name(const) if const else map_dir
    return map_dir, readable


def _register_every_map() -> None:
    """Give ROM_TO_PORYMAP_MAP every map's location name, not just the hand-listed ones.

    The hand-written table above lacks 156 maps (the Safari Zone, Battle Frontier, S.S.
    Tidal, Southern Island, Trick House Puzzle 8, ...); a location missing from it gets
    no map, ASCII or exits. Names come from MapLocation, the
    same enum read_location names maps with, and directories from map_groups.json.
    """
    from pokemon_env.enums import MapLocation

    for group, map_dirs in enumerate(_map_group_dirs()):
        for num, map_dir in enumerate(map_dirs):
            try:
                location = MapLocation((group << 8) | num)
            except ValueError:
                # local_tests/test_map_enum.py fails on this; the server still starts.
                logger.warning(f"MapLocation has no map ({group}, {num}) = {map_dir}")
                continue
            ROM_TO_PORYMAP_MAP.setdefault(location.name.replace("_", " "), map_dir)


@lru_cache(maxsize=64)
def _static_build(map_dir: str) -> tuple:
    """(grid, tiles) of a map from the decomp: the collision grid, one string per row, and
    the tiles walks are planned on, (tile id, behavior, collision, elevation) per cell by
    row (porymap_json_builder raw_tiles: the layout's, also under an ASCII override of the
    same size). (None, ()) when it can't be built."""
    from utils.mapping.porymap_json_builder import build_json_map

    root = _get_pokeemerald_root()
    if not root:
        return None, ()
    try:
        # Every badge: the map's own layout, never a badge-dependent variant
        # (ascii_map_loader CONDITIONAL_SELECTORS). With none, PetalburgCity_Gym came back
        # as the 9x8 lobby numbered from row 0, while its warps are in the 9x112 map's
        # coordinates, so no door of the gym joined anything.
        built = build_json_map(map_dir, root, include_grid=True, include_ascii=False, badge_count=8)
    except Exception as e:  # a broken neighbour must not take the prompt down with it
        logger.warning(f"Porymap: could not build grid for {map_dir}: {e}")
        return None, ()
    grid = (built or {}).get("grid")
    if not grid:
        return None, ()
    tiles = tuple(tuple(tuple(tile) for tile in row) for row in (built.get("raw_tiles") or ()))
    return tuple("".join(str(cell) for cell in row) for row in grid), tiles


def _static_grid(map_dir: str) -> Optional[Tuple[str, ...]]:
    """A map's collision grid from the decomp, one string per row."""
    return _static_build(map_dir)[0]


def _static_tiles(map_dir: str) -> Tuple[Tuple[tuple, ...], ...]:
    """The tiles walks on a map are planned on (_static_build), () when unknown."""
    return _static_build(map_dir)[1]


@lru_cache(maxsize=None)
def layout_tiles(map_dir: str, layout: str = "") -> Tuple[Tuple[tuple, ...], ...]:
    """A map's tiles from the decomp's own layout, (tile id, behavior, collision,
    elevation) per cell by row; () when it can't be read. Never an ASCII override: an
    override draws only walls and floor (RustboroCity's draws its gym door as plain
    floor) and keeps no elevation. `layout` (a LAYOUT_* id) reads that layout instead of
    the map file's: the one a script swapped in (dynamic_map_overlay.swapped_layout)."""
    from utils.mapping.pokeemerald_parser import PokeemeraldLayoutParser, PokeemeraldMapLoader

    root = _get_pokeemerald_root()
    if not root or not map_dir:
        return ()
    try:
        layout = layout or PokeemeraldMapLoader(Path(root)).get_layout_name_from_map(map_dir)
        tiles = PokeemeraldLayoutParser(Path(root)).get_metatiles_with_behavior(layout) if layout else None
    except Exception as e:  # unreadable: callers fall back to what they did without tiles
        logger.warning(f"Porymap: could not read tiles for {map_dir}: {e}")
        return ()
    return tuple(tuple(tuple(tile) for tile in row) for row in tiles or ())


@lru_cache(maxsize=None)
def _warp_tiles(map_dir: str) -> Tuple[Tuple[Optional[tuple], bool], ...]:
    """For each of the map's warp_events, in order: (its layout tile (tile id, behavior,
    collision, elevation) or None, whether a bg_event with a script is on that tile).

    Tiles come from the decomp's own layout, never an ASCII override: an override draws
    only walls and floor, and RustboroCity's draws its gym door as plain floor.
    """
    root = _get_pokeemerald_root()
    map_file = Path(root) / "data" / "maps" / map_dir / "map.json" if root and map_dir else None
    if map_file is None or not map_file.is_file():
        return ()  # an override with its own warps (PetalburgCity_Gym_Lobby) has no map.json
    try:
        with open(map_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:  # unreadable: every warp counts as an ordinary one, as before
        logger.warning(f"Porymap: could not read {map_file}: {e}")
        return ()
    tiles = layout_tiles(map_dir)
    scripted = {
        (int(bg.get("x", -1)), int(bg.get("y", -1))) for bg in data.get("bg_events") or [] if bg.get("script")
    }
    out = []
    for warp in data.get("warp_events") or []:
        x, y = int(warp.get("x", -1)), int(warp.get("y", -1))
        tile = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
        out.append((tuple(tile) if tile is not None and len(tile) > 2 else None, (x, y) in scripted))
    return tuple(out)


@lru_cache(maxsize=None)
def arrival_only_warps(map_dir: str) -> frozenset:
    """Indexes in the map's warp_events of the warps that only mark where a warp arrives.

    pret src/field_control_avatar.c: stepping on a warp event warps only when the tile's
    metatile behavior is a warp one (TryStartWarpEventScript → IsWarpMetatileBehavior);
    arrow warps (TryArrowWarp → IsArrowWarpMetatileBehavior) and doors check their own.
    A warp event on walkable ground with none of those behaviors never fires: it is where
    a warp from elsewhere puts the player. LavaridgeTown_Gym_1F has three, spots B1F's
    geysers throw the player up to (e.g. (3,6)); listed as exits, they drew routes onto
    tiles that go nowhere. Seen live: PETALBURG CITY GYM (6,59)/(7,59), plain floor in
    Alexia's room, fire nothing. A warp in a wall (collision) still counts: doors that
    scripts open at run time (PetalburgCity_Gym's sign doors, sign_door_warps; the Elite
    Four's) are walls in the static layout.
    """
    return frozenset(i for i, (tile, _) in enumerate(_warp_tiles(map_dir)) if lands_only(tile))


def lands_only(tile: Optional[tuple]) -> bool:
    """Whether a warp event on this tile (tile id, behavior, collision, elevation) never
    fires: walkable ground without a warp behavior (arrival_only_warps)."""
    if tile is None or len(tile) < 3:
        return False
    return not tile[2] and str(getattr(tile[1], "name", tile[1])) not in _WARP_BEHAVIORS


# Behaviors a warp event fires on (pret src/metatile_behavior.c): IsWarpMetatileBehavior's
# doors, ladders, escalators, holes, geysers and pads, and IsArrowWarpMetatileBehavior's
# arrows (MetatileBehavior_IsNorthArrowWarp counts MB_STAIRS_OUTSIDE_ABANDONED_SHIP,
# IsSouthArrowWarp MB_WATER_SOUTH_ARROW_WARP and MB_SHOAL_CAVE_ENTRANCE). The water ones
# stay too (Route131's way into Sky Pillar reads as MB_DEEP_WATER here, UnionRoom's way
# out as MB_BRIDGE_OVER_OCEAN): no walk stands on them, and dropping a real one would
# hide it from EXITS.
_WARP_BEHAVIORS = frozenset({
    "ANIMATED_DOOR", "NON_ANIMATED_DOOR", "WATER_DOOR", "DEEP_SOUTH_WARP", "LADDER",
    "UP_ESCALATOR", "DOWN_ESCALATOR", "LAVARIDGE_GYM_B1F_WARP", "LAVARIDGE_GYM_1F_WARP",
    "AQUA_HIDEOUT_WARP", "MT_PYRE_HOLE", "MOSSDEEP_GYM_WARP", "BATTLE_PYRAMID_WARP",
    "CRACKED_FLOOR_HOLE", "EAST_ARROW_WARP", "WEST_ARROW_WARP", "NORTH_ARROW_WARP",
    "SOUTH_ARROW_WARP", "WATER_SOUTH_ARROW_WARP", "STAIRS_OUTSIDE_ABANDONED_SHIP",
    "SHOAL_CAVE_ENTRANCE", "DEEP_WATER", "OCEAN_WATER", "BRIDGE_OVER_OCEAN",
})


# The press that fires an arrow warp, by behavior (pret src/metatile_behavior.c
# MetatileBehavior_Is{North,South,West,East}ArrowWarp).
_ARROW_WARP_PRESS = {
    "NORTH_ARROW_WARP": "UP", "STAIRS_OUTSIDE_ABANDONED_SHIP": "UP",
    "SOUTH_ARROW_WARP": "DOWN", "WATER_SOUTH_ARROW_WARP": "DOWN", "SHOAL_CAVE_ENTRANCE": "DOWN",
    "WEST_ARROW_WARP": "LEFT", "EAST_ARROW_WARP": "RIGHT",
}


def warp_entry(map_dir: str, index: int, x: int, y: int) -> Optional[Tuple[Tuple[int, int], str]]:
    """How warp_events[index] (on tile (x, y)) fires when stepping onto it does not:
    (the tile to stand on, the button to press there); None for a warp that fires on the
    step onto its tile, or whose tile can't be read.

    pret src/field_control_avatar.c ProcessPlayerFieldInput: an arrow warp (TryArrowWarp)
    fires when the player standing on it presses toward its arrow, so a house's exit mat
    is stood on and walked off; an animated door (TryDoorWarp, MetatileBehavior_IsWarpDoor:
    MB_ANIMATED_DOOR) fires when the player on the tile below it presses UP. Every other
    warp (TryStartWarpEventScript: ladders, holes, the Aqua Hideout and Mossdeep Gym pads)
    fires on the step onto its tile.
    """
    tiles = _warp_tiles(map_dir)
    if not 0 <= index < len(tiles) or tiles[index][0] is None:
        return None
    behavior = str(getattr(tiles[index][0][1], "name", tiles[index][0][1]))
    if behavior in _ARROW_WARP_PRESS:
        return (x, y), _ARROW_WARP_PRESS[behavior]
    if behavior == "ANIMATED_DOOR":
        return (x, y + 1), "UP"
    return None


def warp_entry_text(entry: Optional[Tuple[Tuple[int, int], str]], x: int, y: int, offset=(0, 0)) -> str:
    """The EXITS note for a warp_entry: " (stand on it and press DOWN)" or
    " (from (7, 9) press UP into it)", in the walking tools' coordinates (local = map -
    offset), with (x, y) the warp's tile in map coordinates; "" when stepping on the
    warp's tile is enough."""
    if entry is None:
        return ""
    (sx, sy), button = entry
    if (sx, sy) == (x, y):
        return f" (stand on it and press {button})"
    return f" (from ({sx - offset[0]}, {sy - offset[1]}) press {button} into it)"


@lru_cache(maxsize=None)
def sign_door_warps(map_dir: str) -> frozenset:
    """Indexes in the map's warp_events of doors gone through by reading them: the warp's
    tile can't be walked onto (collision) and a bg_event with a script stands on it.

    pret data/maps/PetalburgCity_Gym: each room door is a warp whose tile has a sign
    (BG_EVENT_PLAYER_FACING_ANY), e.g. PetalburgCity_Gym_EventScript_SpeedRoomDoor at
    (1,105): MSGBOX_YESNO "…ROOM,' the sign says. Do you want to go through?", YES →
    warpdoor MAP_PETALBURG_CITY_GYM 7,85 (warp 3's tile); a locked door says "The door
    appears to be locked." Seen live: walking up into (7,79) is BLOCKED; A, then YES, went
    through.
    """
    return frozenset(
        i for i, (tile, scripted) in enumerate(_warp_tiles(map_dir))
        if scripted and tile is not None and tile[2]
    )


def edge_blocked(surf: bool = False) -> set:
    """_EDGE_BLOCKED, less water when the party can Surf: a surfer crosses a map edge
    over water like a walker over ground (the sea routes east of Lilycove)."""
    return set(_EDGE_BLOCKED) - ({"W"} if surf else set())


def connection_open_span(conn: Dict[str, Any], surf: bool = False) -> Optional[set]:
    """Positions along this map's edge where the connected map is walkable (or, with
    surf, surfable).

    A map edge only leads somewhere where the tile across it, on the neighbouring map,
    can be stood on: Oldale Town's whole south row is open ground, but Route 101 has
    trees under all of it except x=8-11. Positions are in this map's frame (x for a
    north/south edge, y for east/west), using pokeemerald's connection offset
    (neighbour = this - offset). None when the neighbour's layout can't be read.
    """
    direction = conn.get("direction")
    map_dir = _map_dirs_by_const().get(str(conn.get("map", "")))
    grid = _static_grid(map_dir) if map_dir else None
    if direction not in ("up", "down", "left", "right") or not grid:
        return None
    blocked = edge_blocked(surf)
    offset = int(conn.get("offset", 0) or 0)
    if direction in ("up", "down"):
        row = grid[-1] if direction == "up" else grid[0]
        return {i + offset for i, cell in enumerate(row) if cell not in blocked}
    width = max(len(row) for row in grid)
    col = width - 1 if direction == "left" else 0
    return {
        y + offset for y, row in enumerate(grid)
        if col < len(row) and row[col] not in blocked
    }


def _edge_openings(ascii_lines: List[str], direction: str, allowed: Optional[set] = None, surf: bool = False) -> str:
    """Walkable stretches on one edge of the map, e.g. 'x=10-11, y=0'.

    allowed, when given, keeps only those positions along the edge (see
    connection_open_span). surf: water counts as open (edge_blocked).
    """
    if not ascii_lines:
        return ""
    height, width = len(ascii_lines), max(len(line) for line in ascii_lines)
    if direction in ("up", "down"):
        y = 0 if direction == "up" else height - 1
        cells = [(x, y) for x in range(len(ascii_lines[y]))]
    elif direction in ("left", "right"):
        x = 0 if direction == "left" else width - 1
        cells = [(x, y) for y in range(height) if x < len(ascii_lines[y])]
    else:
        return ""
    axis = 0 if direction in ("up", "down") else 1
    blocked = edge_blocked(surf)
    open_cells = [
        c for c in cells
        if ascii_lines[c[1]][c[0]] not in blocked and (allowed is None or c[axis] in allowed)
    ]
    if not open_cells:
        return ""
    # Group consecutive open tiles into ranges along the edge.
    ranges, start, prev = [], open_cells[0][axis], open_cells[0][axis]
    for cell in open_cells[1:]:
        if cell[axis] != prev + 1:
            ranges.append((start, prev))
            start = cell[axis]
        prev = cell[axis]
    ranges.append((start, prev))
    spans = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)
    if axis == 0:
        return f"x={spans}, y={open_cells[0][1]}"
    return f"x={open_cells[0][0]}, y={spans}"


_TRAINER_FLAGS_START = 0x500  # pret include/constants/flags.h


@lru_cache(maxsize=1)
def _trainer_ids() -> Dict[str, int]:
    """TRAINER_JOSH -> 600 etc., from include/constants/opponents.h (copied from pret)."""
    root = _get_pokeemerald_root()
    path = Path(root) / "include" / "constants" / "opponents.h" if root else None
    if path is None or not path.exists():
        return {}
    ids = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("TRAINER_") and parts[2].isdigit():
            ids[parts[1]] = int(parts[2])
    return ids


@lru_cache(maxsize=64)
def _map_trainers(map_dir: str) -> Tuple[Tuple[int, int, str, int], ...]:
    """(x, y, name, trainer id) for each object on a map whose script starts a trainer battle.

    The object's own trainer_type isn't enough: gym leaders are TRAINER_TYPE_NONE and
    battle from their script. The script label's first trainerbattle line names the
    trainer (`trainerbattle_single TRAINER_JOSH, ...`).
    """
    root = _get_pokeemerald_root()
    if not root or not map_dir:
        return ()
    map_path = Path(root) / "data" / "maps" / map_dir
    try:
        with open(map_path / "map.json", encoding="utf-8") as fh:
            objects = json.load(fh).get("object_events") or []
        scripts = (map_path / "scripts.inc").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ()
    # Floors of one place can keep their scripts in a sibling's file: pret has
    # LavaridgeTown_Gym_B1F_EventScript_Jace in LavaridgeTown_Gym_1F/scripts.inc.
    stem = map_dir.rpartition("_")[0]
    if stem:
        for sibling in sorted(map_path.parent.glob(f"{stem}_*/scripts.inc")):
            if sibling.parent != map_path:
                try:
                    scripts += "\n" + sibling.read_text(encoding="utf-8")
                except OSError:
                    continue
    ids = _trainer_ids()
    out = []
    for obj in objects:
        label = f"{obj.get('script', '')}::"
        start = scripts.find(label)
        if not obj.get("script") or start == -1:
            continue
        body_end = scripts.find("::", start + len(label))
        body = scripts[start + len(label): body_end if body_end != -1 else len(scripts)]
        for line in body.splitlines():
            words = line.replace(",", " ").split()
            # The trainer comes first (`trainerbattle_single TRAINER_JOSH, ...`), or after
            # the battle type in the generic macro (pret LavaridgeTown_Gym_1F: `trainerbattle
            # TRAINER_BATTLE_CONTINUE_SCRIPT, TRAINER_COLE, ...`), whose gym trainers were
            # all missing here.
            trainer = next((word for word in words[1:3] if word in ids), None)
            if words and words[0].startswith("trainerbattle") and trainer:
                out.append((int(obj.get("x", 0)), int(obj.get("y", 0)), trainer_name(trainer), ids[trainer]))
                break
    return tuple(out)


def trainer_name(trainer_const: str) -> str:
    """TRAINER_TATE_AND_LIZA_1 -> TATE AND LIZA: the number is the rematch tier."""
    name = trainer_const[len("TRAINER_"):] if trainer_const.startswith("TRAINER_") else trainer_const
    head, _, tail = name.rpartition("_")
    if head and tail.isdigit():
        name = head
    return name.replace("_", " ")


@lru_cache(maxsize=1)
def trainer_names() -> Dict[Tuple[str, ...], Tuple[int, ...]]:
    """("TATE", "AND", "LIZA") -> every trainer id under that name (pret opponents.h; the
    rematch tiers TRAINER_TATE_AND_LIZA_1..5 share one name)."""
    names: Dict[Tuple[str, ...], List[int]] = {}
    for const, trainer_id in _trainer_ids().items():
        if trainer_id:  # TRAINER_NONE
            names.setdefault(tuple(trainer_name(const).split()), []).append(trainer_id)
    return {name: tuple(ids) for name, ids in names.items()}


def _script_labels(text: str) -> Dict[str, List[str]]:
    """label -> its lines, for a pret scripts.inc: a script starts at a `Label::` line (a
    `Label:` local one too) at the start of a line and runs to the next."""
    labels: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and not line[:1].isspace() and stripped.rstrip(":") and " " not in stripped:
            current = stripped.rstrip(":")
            labels[current] = []
        elif current is not None:
            labels[current].append(stripped)
    return labels


def _map_script_labels(map_dir: str) -> Dict[str, List[str]]:
    """_script_labels of map_dir's scripts.inc and its sibling floors' (pret keeps
    LavaridgeTown_Gym_B1F's scripts in LavaridgeTown_Gym_1F/scripts.inc)."""
    root = _get_pokeemerald_root()
    if not root or not map_dir:
        return {}
    map_path = Path(root) / "data" / "maps" / map_dir
    paths = [map_path / "scripts.inc"]
    stem = map_dir.rpartition("_")[0]
    if stem:
        paths += [p for p in sorted(map_path.parent.glob(f"{stem}_*/scripts.inc")) if p.parent != map_path]
    labels: Dict[str, List[str]] = {}
    for path in paths:
        try:
            labels.update(_script_labels(path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return labels


def _badge_set_in(lines: List[str]) -> Optional[int]:
    """N of the first `setflag FLAG_BADGE0N_GET` among lines, else None."""
    for line in lines:
        words = line.split()
        if len(words) == 2 and words[0] == "setflag" and words[1].startswith("FLAG_BADGE") and words[1].endswith("_GET"):
            number = words[1][len("FLAG_BADGE"):-len("_GET")]
            if number.isdigit():
                return int(number)
    return None


def _battle_trainer(line: str, ids: Dict[str, int]) -> Optional[str]:
    """The TRAINER_ constant a `trainerbattle...` line starts a battle with, else None
    (it comes first, or after the battle type in the generic macro: _map_trainers)."""
    words = line.replace(",", " ").split()
    if not words or not words[0].startswith("trainerbattle"):
        return None
    return next((word for word in words[1:3] if word in ids), None)


@lru_cache(maxsize=1)
def gym_leaders() -> Dict[int, Tuple[str, int, str]]:
    """trainer id -> (name, badge number, map dir) of every trainer whose win gives a badge.

    Read from pret's map scripts, not listed by hand: a gym leader's trainerbattle names
    the script that runs after the win, and that script sets the badge
    (MossdeepCity_Gym_EventScript_TateAndLizaDefeated: `setflag FLAG_BADGE07_GET`); Norman's
    win runs on in the same script (PetalburgCity_Gym_EventScript_NormanBattle).
    """
    root = _get_pokeemerald_root()
    if not root:
        return {}
    ids = _trainer_ids()
    leaders: Dict[int, Tuple[str, int, str]] = {}
    for path in sorted((Path(root) / "data" / "maps").glob("*/scripts.inc")):
        try:
            labels = _script_labels(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for body in labels.values():
            for line in body:
                trainer = _battle_trainer(line, ids)
                if trainer is None or ids[trainer] in leaders:
                    continue
                after_win = [labels[word] for word in line.replace(",", " ").split() if word in labels]
                badge = next((b for b in map(_badge_set_in, [body, *after_win]) if b), None)
                if badge:
                    leaders[ids[trainer]] = (trainer_name(trainer), badge, path.parent.name)
    return leaders


@lru_cache(maxsize=64)
def gym_leader_objects(map_dir: str) -> Tuple[Tuple[int, int, str, int, int], ...]:
    """(x, y, name, trainer id, badge number) of each object on map_dir whose script starts
    a gym leader's battle (gym_leaders), by itself or through the scripts it goes to or
    calls: pret PetalburgCity_Gym_EventScript_Norman picks NormanBattle by
    VAR_PETALBURG_GYM_STATE. Map-file tiles: the leaders never move."""
    leaders = gym_leaders()
    labels = _map_script_labels(map_dir)
    root = _get_pokeemerald_root()
    if not leaders or not labels or not root:
        return ()
    try:
        with open(Path(root) / "data" / "maps" / map_dir / "map.json", encoding="utf-8") as fh:
            objects = json.load(fh).get("object_events") or []
    except (OSError, ValueError):
        return ()
    ids = _trainer_ids()
    out = []
    for obj in objects:
        seen, todo = set(), [str(obj.get("script") or "")]
        found = None
        while todo and found is None:
            label = todo.pop()
            if label in seen or label not in labels:
                continue
            seen.add(label)
            for line in labels[label]:
                trainer = _battle_trainer(line, ids)
                if trainer is not None and ids[trainer] in leaders:
                    found = ids[trainer]
                    break
                todo += [word for word in line.replace(",", " ").split() if word in labels]
        if found is not None:
            name, badge, _ = leaders[found]
            out.append((int(obj.get("x", 0)), int(obj.get("y", 0)), name, found, badge))
    return tuple(out)


def _format_trainers(map_dir: str, memory_reader: Any) -> List[str]:
    """Trainers on this map the player hasn't beaten yet, with where they stand.

    Seen live: objective "Defeat Gym trainers" in Rustboro Gym. The model crept up one tile
    per decision, not knowing where anyone was. Beaten is read from the game's trainer
    flags (TRAINER_FLAGS_START + id), so the list shrinks as they fall.
    """
    trainers = _map_trainers(map_dir)
    if not trainers or memory_reader is None:
        return []
    left = []
    for x, y, name, trainer_id in trainers:
        try:
            beaten = memory_reader._is_event_flag_set(_TRAINER_FLAGS_START + trainer_id)
        except Exception:
            return []  # can't read flags: say nothing rather than something wrong
        if not beaten:
            left.append(f"{name} at ({x}, {y})")
    if not left:
        return ["- every trainer on this map is already beaten"]
    # Seen live in Rustboro Gym: the model went from trainer to trainer with a worn-down
    # Torchic. The objectives say where to heal; this keeps the warning next to the names.
    try:
        weak = bool(memory_reader.read_team_condition()["weak"])
    except Exception:
        weak = False  # a warning that can't be read is left out, never guessed
    return [
        "- " + "; ".join(left) + (" (heal first: team is weak)" if weak else ""),
        "  (walking into a trainer's line of sight starts the battle; or stand next to them, face them and press A)",
    ]


def _format_rematches(map_dir: str, memory_reader: Any) -> List[str]:
    """Match Call rematches (server.rematch): who wants a rematch and where, and how this
    map's unregistered rematch trainers get registered. [] without Match Call, without a
    server, or when RAM can't be read.

    Asked for (2026-11-05): with six badges the live team had ten registered trainers
    waiting for a rematch (trainer battles pay 1.5x EXP, pret Cmd_getexp) and nothing in
    the state said so."""
    if memory_reader is None:
        return []
    try:
        from server import rematch
    except ImportError:
        return []
    try:
        state = rematch.read_rematches(memory_reader)
    except Exception as e:  # a line that can't be read is left out, never guessed
        logger.debug(f"rematches unreadable: {e}")
        return []
    return [line for line in (rematch.ready_line(state, map_dir), rematch.registration_hint(state, map_dir)) if line]


def _party_can_surf() -> bool:
    """server.game_tools.can_surf (the party knows SURF and has the Balance Badge), or
    False where no server is running (its readers are installed by server/app.py)."""
    try:
        from server import game_tools
    except ImportError:
        return False
    return game_tools.can_surf()


def _party_can_push_boulders() -> bool:
    """server.game_tools.can_push_boulders (STRENGTH known, the Heat Badge), or False where
    no server is running."""
    try:
        from server import game_tools
    except ImportError:
        return False
    return game_tools.can_push_boulders()


def _boulder_places(map_dir: Optional[str], memory_reader: Any) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """{map-file tile: tile it stands on now} for the map's Strength boulders
    (server.game_tools.boulder_places, read live from gObjectEvents). Before, MAGMA
    HIDEOUT 1F's boulders were drawn as people ('N') on their map-file tiles, pushed or
    not (local_tests/test_strength.py). {} without boulders or without a server."""
    try:
        from server import game_tools
        from server.field_control import loaded_boulders
    except ImportError:
        return {}
    if not map_dir:
        return {}
    loaded = None
    if memory_reader is not None:
        try:
            loaded = loaded_boulders(memory_reader)
        except Exception as e:  # unreadable: the map-file tiles
            logger.debug(f"live boulders unreadable: {e}")
    return game_tools.boulder_places(map_dir, loaded if loaded is not None else {})


def _format_exits(json_map: Dict[str, Any], ascii_lines: List[str], surf: bool = False) -> List[str]:
    """Plain-language list of where each exit leads.

    The same facts are in the Map Data JSON below, but a small model reads the raw
    warp/connection objects poorly (e.g. it took Birch's lab door for the way to
    Route 101, which is an open map edge). One line per exit is cheap and unambiguous.

    surf (the party can Surf): an edge over water counts as a way off, and says so.
    """
    lines = []
    for conn in json_map.get("connections", []):
        direction = conn.get("direction", "?")
        span = connection_open_span(conn, surf)
        where = _edge_openings(ascii_lines, direction, span, surf)
        on_foot = _edge_openings(ascii_lines, direction, connection_open_span(conn), False) if surf else where
        edge = {"up": "north", "down": "south", "left": "west", "right": "east"}.get(direction, direction)
        if where and where != on_foot:
            how = f" (walk or Surf off at {where}; go_to_exit surfs where it must)"
        elif where:
            how = f" (walk off at {where})"
        elif span is not None and ascii_lines:
            # Seen live: Route 104's "south edge -> ROUTE 105" (all sea) drew the model
            # south when it had no way to cross.
            how = " (no way across: cliff all along it)" if surf else " (no way across on foot: water or cliff all along it)"
        else:
            how = ""
        lines.append(f"- {edge} edge of the map{how} -> {_readable_map_name(conn.get('map', '?'))}")
    # Warps in the map's own order (an override with its own warps has no map.json of
    # that name, so none are left out there).
    map_dir = str(json_map.get("name") or "")
    arrival_only = arrival_only_warps(map_dir)
    for i, warp in enumerate(json_map.get("warps", [])):
        if i in arrival_only:
            continue  # only where a warp from elsewhere lands; stepping on it does nothing
        x, y = warp.get("x", "?"), warp.get("y", "?")
        how = warp_entry_text(warp_entry(map_dir, i, x, y), x, y) if isinstance(x, int) and isinstance(y, int) else ""
        lines.append(f"- door/warp at ({x}, {y}) -> {_readable_map_name(warp.get('dest_map', '?'))}{how}")
    return lines


def _format_porymap_info(
    location_name: Optional[str],
    player_coords: Optional[Tuple[int, int]] = None,
    badge_count: int = 0,
    memory_reader: Any = None,
    runtime_object_events: Optional[List[Dict[str, Any]]] = None,
) -> PorymapResult:
    """
    Format porymap ground truth data (JSON and ASCII map) for the agent.
    
    Args:
        location_name: Current location name from ROM
        player_coords: Player's (x, y) coordinates
        badge_count: Number of badges player has (for game-state-aware map selection)
        memory_reader: Optional PokemonEmeraldReader for live metatile reads on dynamic maps
    
    Returns a PorymapResult containing formatted strings and optional raw map data.
    """
    context_parts = []
    
    if not location_name or location_name == 'TITLE_SEQUENCE' or location_name == 'Unknown':
        return PorymapResult(context_parts=context_parts)
    
    try:
        # Import here to avoid circular dependencies and allow graceful failure
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.mapping.pokeemerald_parser import PokeemeraldMapLoader
        
        # Get pokeemerald root
        pokeemerald_root = _get_pokeemerald_root()
        if not pokeemerald_root:
            logger.warning(f"Porymap: Could not find pokeemerald root for location '{location_name}'")
            return PorymapResult(context_parts=context_parts)
        
        # Convert ROM location name to porymap map name using mapping
        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

        # Debug log for Glass Workshop issue
        if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
            logger.warning(f"DEBUG: Glass Workshop mapping - ROM location: '{location_name}' -> porymap: '{porymap_map_name}'")

        # If not in direct mapping, try fuzzy matching
        if not porymap_map_name:
            map_loader = PokeemeraldMapLoader(pokeemerald_root)
            
            def normalize_for_matching(name: str) -> str:
                """Normalize location name for fuzzy matching."""
                # Normalize: lowercase, remove spaces/underscores, remove common suffixes
                normalized = str(name).lower().replace(" ", "").replace("_", "").replace("town", "").replace("city", "").replace("route", "")
                # Also try removing "professor", "birchs", "lab" for building matching
                if "professor" in normalized or "birch" in normalized or "lab" in normalized:
                    normalized = normalized.replace("professor", "").replace("birchs", "").replace("birch", "").replace("lab", "")
                return normalized
            
            rom_normalized = normalize_for_matching(location_name)
            maps_dir = pokeemerald_root / "data" / "maps"

            # Debug log for Glass Workshop fuzzy matching
            if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
                logger.warning(f"DEBUG: Fuzzy matching Glass Workshop - normalized: '{rom_normalized}'")

            if maps_dir.exists():
                # Try direct directory name match
                best_match = None
                best_match_score = 0

                for map_dir in maps_dir.iterdir():
                    if not map_dir.is_dir() or map_dir.name == "map_groups.json":
                        continue
                    
                    map_name = map_dir.name
                    map_normalized = normalize_for_matching(map_name)
                    
                    # Exact match
                    if rom_normalized == map_normalized:
                        porymap_map_name = map_name
                        logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via fuzzy match")
                        break
                    
                    # Partial match scoring (for cases like "LITTLEROOT TOWN PROFESSOR BIRCHS LAB" -> "LittlerootTown_ProfessorBirchsLab")
                    if rom_normalized in map_normalized or map_normalized in rom_normalized:
                        match_length = min(len(rom_normalized), len(map_normalized))
                        if match_length > best_match_score and match_length > 5:  # Require at least 5 chars match
                            best_match = map_name
                            best_match_score = match_length
                
                # Use best partial match if no exact match found
                if not porymap_map_name and best_match:
                    porymap_map_name = best_match
                    logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via partial match (score: {best_match_score})")
        
        if not porymap_map_name:
            logger.warning(f"Porymap: Could not map ROM location '{location_name}' to porymap map name")
            return PorymapResult(context_parts=context_parts)
        
        logger.info(f"Porymap: Building map for '{porymap_map_name}' (ROM location: '{location_name}', badges: {badge_count})")
        
        # Build JSON map (with grid included for pathfinding, even though we don't show it in prompt)
        # Pass badge_count for game-state-aware map selection (e.g., Petalburg Gym lobby)
        try:
            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root, badge_count=badge_count)
        except ValueError as e:
            logger.error(f"Porymap: Failed to build map for '{porymap_map_name}' due to corrupted tileset data: {e}")
            logger.error("This likely indicates missing or corrupted tileset files in pokemon_env/porymap.")
            logger.error("Pathfinding will not be available for this location.")
            return PorymapResult(context_parts=context_parts)
        
        # Ensure grid is built (even if we don't include it in the text output)
        if not json_map.get('grid'):
            # Rebuild with grid if needed
            from utils.mapping.porymap_json_builder import build_json_map
            try:
                json_map_with_grid = build_json_map(porymap_map_name, pokeemerald_root, include_grid=True, include_ascii=True)
                if json_map_with_grid and json_map_with_grid.get('grid'):
                    json_map['grid'] = json_map_with_grid['grid']
            except ValueError as e:
                logger.error(f"Porymap: Failed to rebuild grid for '{porymap_map_name}': {e}")

        if not json_map:
            logger.warning(f"Porymap: Failed to build JSON map for '{porymap_map_name}'")
            return PorymapResult(context_parts=context_parts)

        # For dynamic maps (e.g. Mauville Gym), replace static grid with live
        # emulator metatiles so barrier changes are reflected in the agent's map.
        if memory_reader is not None:
            from utils.mapping.dynamic_map_overlay import apply_live_overlay_to_json_map
            apply_live_overlay_to_json_map(json_map, memory_reader, location_name)

        # Filter grid based on player elevation to handle multi-level maps
        # For caves/dungeons with multiple connected levels, be more permissive
        # For buildings/bridges with truly separate floors, be strict
        if player_coords and json_map.get('raw_tiles') and json_map.get('grid'):
            try:
                px, py = player_coords[0], player_coords[1]
                raw_tiles = json_map['raw_tiles']

                # Get player's current elevation from the tile they're standing on
                if 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
                    player_tile = raw_tiles[py][px]
                    if len(player_tile) >= 4:
                        player_elevation = player_tile[3]  # elevation is 4th element

                        # Check if this is a cave/dungeon (has elevation variety but connected paths)
                        # vs a multi-floor building (strict separation)
                        elevations_in_map = set()
                        for row in raw_tiles:
                            for tile in row:
                                if len(tile) >= 4:
                                    elevations_in_map.add(tile[3])

                        grid = json_map['grid']
                        height = len(grid)
                        width = len(grid[0]) if height > 0 else 0

                        # Flood-fill from the player over (tile, the walker's elevation
                        # there) by the walks' own step rules (pathfinding.walk_steps: pret
                        # IsElevationMismatchAt and ObjectEventUpdateElevation, collision,
                        # Surf). Ledges are entered from any side: the far side stays drawn.
                        # Seen live on MOSSDEEP CITY: (16,7), elevation 5, was drawn '.'
                        # beside the player on the west shore (16,6), elevation 3, and the
                        # walk into it was refused every time.
                        from collections import deque
                        from utils.mapping.pathfinding import WATER_ELEVATION, walk_steps

                        def _get_elev(tx, ty):
                            if 0 <= ty < len(raw_tiles) and 0 <= tx < len(raw_tiles[ty]):
                                t = raw_tiles[ty][tx]
                                return t[3] if len(t) >= 4 else None
                            return None

                        walkable_chars = {'.', '~', 'S', 'D', '&', 'W', '←', '→', '↑', '↓'}
                        walls = {ch for row in grid for ch in row} - walkable_chars
                        player_char = grid[py][px] if 0 <= py < height and 0 <= px < width else '.'
                        surf = player_char == 'W' or _party_can_surf()
                        # The player's own elevation (on a bridge the tile can't tell it).
                        start_elevation = player_elevation if player_elevation != 15 else None
                        if memory_reader is not None:
                            try:
                                from server.field_control import player_elevation as live_elevation

                                live = live_elevation(memory_reader)
                                start_elevation = live if player_elevation == 15 else player_elevation
                            except Exception:
                                pass  # the tile's elevation stands in
                        reachable = {(px, py)}
                        seen = {(px, py, start_elevation)}
                        queue = deque([(px, py, start_elevation)])
                        while queue:
                            cx, cy, walker = queue.popleft()
                            for nx, ny, after in walk_steps(raw_tiles, grid, cx, cy, walker, surf, walls):
                                if (nx, ny, after) in seen:
                                    continue
                                seen.add((nx, ny, after))
                                reachable.add((nx, ny))
                                queue.append((nx, ny, after))

                        def _other_floor(x, y):
                            """Floor the player can't step onto from their own: another
                            elevation by pret IsElevationMismatchAt (a tile at 0 or 15
                            matches any). Not while surfing: the shore is landed on."""
                            here = start_elevation
                            there = _get_elev(x, y)
                            if here in (None, 0, WATER_ELEVATION) or there is None:
                                return False
                            return there not in (0, 15) and there != here

                        # Build filtered grid: floor the walk can't reach from here, or
                        # can't step onto from the player's floor (another elevation, joined
                        # only by stairs, a ladder or a ramp at elevation 0), becomes '^'.
                        filtered_grid = []
                        for y in range(height):
                            filtered_row = []
                            for x in range(width):
                                original_char = grid[y][x]

                                # Always preserve special markers regardless of reachability
                                if original_char in ['D', 'S', 'T', 'K', 'N', 'I', 'P', 'V']:
                                    filtered_row.append(original_char)
                                elif original_char == '#':
                                    filtered_row.append('#')
                                elif (x, y) in reachable and original_char in ('.', '~', '&') and _other_floor(x, y):
                                    filtered_row.append('^')
                                elif (x, y) in reachable:
                                    # Handle bridge tiles: if player is under a bridge, show as walkable ground
                                    if original_char == '&':
                                        tile_elev = _get_elev(x, y)
                                        if tile_elev is not None and tile_elev > player_elevation:
                                            filtered_row.append('.')
                                        else:
                                            filtered_row.append('&')
                                    else:
                                        filtered_row.append(original_char)
                                elif original_char in ['.', '~']:
                                    # Walkable ground at a different elevation
                                    filtered_row.append('^')
                                elif original_char == 'W':
                                    # Unreachable water stays '#' (need Surf, not elevation),
                                    # unless the party can Surf: then the walks cross it.
                                    player_char = grid[py][px] if 0 <= py < height and 0 <= px < width else '.'
                                    if player_char == 'W' or _party_can_surf():
                                        filtered_row.append('W')
                                    else:
                                        filtered_row.append('#')
                                else:
                                    # Anything else unreachable becomes '#'
                                    filtered_row.append('#')
                            filtered_grid.append(filtered_row)

                        # Update the grid and ASCII map
                        json_map['grid'] = filtered_grid

                        # Regenerate ASCII from filtered grid unless this map uses override ASCII
                        # (override ASCII must be kept verbatim so P/V/K/S/I/D match the override)
                        if not json_map.get('ascii_from_override'):
                            ascii_lines = [''.join(row) for row in filtered_grid]
                            json_map['ascii'] = '\n'.join(ascii_lines)
                        elif json_map.get('ascii'):
                            # Override ASCII keeps its own letters; only its floor that is
                            # '^' above (another floor, or out of reach) is redrawn, as on
                            # any other map (FORTREE CITY's treetops from the ground,
                            # MOSSDEEP CITY's upper town from the west shore).
                            drawn = json_map['ascii'].split('\n')
                            if len(drawn) == height and all(len(line) == width for line in drawn):
                                json_map['ascii'] = '\n'.join(
                                    ''.join(
                                        '^' if ch in ('.', '~', '&') and filtered_grid[y][x] == '^' else ch
                                        for x, ch in enumerate(line)
                                    )
                                    for y, line in enumerate(drawn)
                                )

                        # Count how many tiles were blocked by elevation filtering
                        blocked_count = sum(1 for row in filtered_grid for cell in row if cell == '#')
                        original_blocked_count = sum(1 for row in grid for cell in row if cell == '#')
                        newly_blocked = blocked_count - original_blocked_count

                        logger.info(f"Elevation filtering (flood-fill): player at ({px}, {py}) elevation {player_elevation} - map has elevations {sorted(elevations_in_map)} - reachable tiles: {len(reachable)} - blocked {newly_blocked} additional tiles")
            except Exception as e:
                logger.warning(f"Failed to filter map by elevation: {e}")

        # Each object on the tile it stands on now (field_control.object_places), before
        # anything below compares tiles: seen live in MOSSDEEP CITY GYM, a statue the floor
        # switch had moved to (4,20) was drawn on its map-file tile (5,21).
        if memory_reader is not None and json_map.get("objects"):
            try:
                from server.field_control import object_places

                json_map["objects"] = objects_where_they_stand(json_map["objects"], object_places(memory_reader))
            except Exception as e:  # unreadable: the map-file tiles, as before
                logger.debug(f"object places unavailable: {e}")

        # Reconcile static porymap objects against runtime object events.
        npc_source = "static_reference"
        static_objects = json_map.get("objects", [])
        # region agent log
        _agent_debug_log(
            "H3",
            "porymap_state.py:_format_porymap_info:pre_reconcile",
            "Pre-reconciliation object snapshot",
            {
                "location_name": location_name,
                "porymap_map_name": porymap_map_name,
                "static_count": len(static_objects),
                "runtime_count": len(runtime_object_events or []),
                "static_local_ids": [obj.get("local_id") for obj in static_objects[:10]],
                "runtime_local_ids": [obj.get("local_id") for obj in (runtime_object_events or [])[:10]],
                "runtime_coords": [
                    {
                        "local_id": obj.get("local_id"),
                        "x": obj.get("current_x", obj.get("x")),
                        "y": obj.get("current_y", obj.get("y")),
                        "graphics_id": obj.get("graphics_id"),
                    }
                    for obj in (runtime_object_events or [])[:10]
                ],
            },
        )
        # endregion
        if runtime_object_events:
            reconciled_objects: List[Dict[str, Any]] = []
            runtime_with_coords: List[Dict[str, Any]] = []
            match_debug: List[Dict[str, Any]] = []

            for runtime_obj in runtime_object_events:
                rx, ry = _get_runtime_xy(runtime_obj)
                if isinstance(rx, int) and isinstance(ry, int):
                    runtime_with_coords.append(runtime_obj)

            had_runtime_coords = len(runtime_with_coords) > 0

            for static_obj in static_objects:
                static_local_id = _normalize_local_id(static_obj.get("local_id"))
                static_x = static_obj.get("x")
                static_y = static_obj.get("y")
                static_gid = static_obj.get("graphics_id")

                matched = False
                matched_reason = ""
                matched_runtime: Optional[Dict[str, Any]] = None

                matched_idx = -1

                # Primary: local_id matching when both sides provide one.
                if static_local_id:
                    for idx, runtime_obj in enumerate(runtime_with_coords):
                        runtime_local_id = _normalize_local_id(runtime_obj.get("local_id"))
                        if runtime_local_id and runtime_local_id == static_local_id:
                            matched = True
                            matched_reason = "local_id"
                            matched_runtime = runtime_obj
                            matched_idx = idx
                            break

                # Fallback: coordinate proximity + compatible graphics.
                if not matched and isinstance(static_x, int) and isinstance(static_y, int):
                    for idx, runtime_obj in enumerate(runtime_with_coords):
                        rx, ry = _get_runtime_xy(runtime_obj)
                        if not isinstance(rx, int) or not isinstance(ry, int):
                            continue
                        if abs(rx - static_x) + abs(ry - static_y) > 2:
                            continue
                        if not _graphics_ids_compatible(static_gid, runtime_obj.get("graphics_id")):
                            continue
                        matched = True
                        matched_reason = "coord_plus_graphics"
                        matched_runtime = runtime_obj
                        matched_idx = idx
                        break

                if matched and matched_idx >= 0:
                    runtime_with_coords.pop(matched_idx)

                if len(match_debug) < 12:
                    rx, ry = _get_runtime_xy(matched_runtime) if matched_runtime else (None, None)
                    match_debug.append(
                        {
                            "static_local_id": static_obj.get("local_id"),
                            "static_flag": static_obj.get("flag"),
                            "static_x": static_x,
                            "static_y": static_y,
                            "static_graphics_id": static_gid,
                            "matched": matched,
                            "matched_reason": matched_reason,
                            "runtime_source": (matched_runtime or {}).get("source"),
                            "runtime_local_id": (matched_runtime or {}).get("local_id"),
                            "runtime_graphics_id": (matched_runtime or {}).get("graphics_id"),
                            "runtime_x": rx,
                            "runtime_y": ry,
                        }
                    )

                if matched:
                    reconciled_objects.append(static_obj)

            # Only switch to runtime-verified mode when we had runtime coords to compare.
            if had_runtime_coords:
                json_map["objects"] = reconciled_objects
                npc_source = "runtime_verified"
        # region agent log
        _agent_debug_log(
            "H4",
            "porymap_state.py:_format_porymap_info:post_reconcile",
            "Post-reconciliation object snapshot",
            {
                "location_name": location_name,
                "npc_source": npc_source,
                "final_count": len(json_map.get("objects", [])),
                "final_local_ids": [obj.get("local_id") for obj in json_map.get("objects", [])[:10]],
                "birch_present": any(obj.get("local_id") == "LOCALID_BIRCHS_LAB_BIRCH" for obj in json_map.get("objects", [])),
            },
        )
        # endregion
        # region agent log
        _agent_debug_log(
            "H10",
            "porymap_state.py:_format_porymap_info:match_debug",
            "Per-object reconciliation decision trace",
            {
                "location_name": location_name,
                "match_debug": match_debug if "match_debug" in locals() else [],
            },
        )
        # endregion

        pre_flag_count = len(json_map.get("objects", []))
        if memory_reader is not None and json_map.get("objects"):
            filtered_by_flag: List[Dict[str, Any]] = []
            flag_filter_debug: List[Dict[str, Any]] = []
            for obj in json_map["objects"]:
                flag_name = obj.get("flag", "0")
                if flag_name and flag_name != "0":
                    flag_id = FLAG_HIDE_IDS.get(flag_name)
                    if flag_id is not None:
                        try:
                            is_set = memory_reader._is_event_flag_set(flag_id)
                        except Exception:
                            is_set = False
                        if len(flag_filter_debug) < 12:
                            flag_filter_debug.append({"flag_name": flag_name, "flag_id": flag_id, "is_set": is_set, "local_id": obj.get("local_id", ""), "graphics_id": obj.get("graphics_id", "")})
                        if is_set:
                            continue
                    else:
                        if len(flag_filter_debug) < 12:
                            flag_filter_debug.append({"flag_name": flag_name, "flag_id": None, "is_set": None, "local_id": obj.get("local_id", ""), "graphics_id": obj.get("graphics_id", ""), "note": "unknown_flag"})
                filtered_by_flag.append(obj)
            json_map["objects"] = filtered_by_flag
            npc_source = npc_source + "+flag_filtered" if npc_source != "static_reference" else "flag_filtered"
            # region agent log
            _agent_debug_log(
                "H12",
                "porymap_state.py:_format_porymap_info:flag_filter",
                "Flag-based object filtering",
                {
                    "location_name": location_name,
                    "pre_flag_count": pre_flag_count,
                    "post_flag_count": len(filtered_by_flag),
                    "removed_count": pre_flag_count - len(filtered_by_flag),
                    "flag_filter_debug": flag_filter_debug,
                },
            )
            # endregion
        
        # Format for LLM
        context_parts.append("\n=== PORYMAP MAP LAYOUT ===")
        context_parts.append(f"Location: {json_map.get('name', porymap_map_name)}")
        context_parts.append(f"Dimensions: {json_map['dimensions']['width']}x{json_map['dimensions']['height']}")
        
        # Strength boulders: on the tile they stand on now, not their map-file tile.
        places = _boulder_places(porymap_map_name, memory_reader)
        if places and json_map.get("objects"):
            json_map["objects"] = [
                {**obj, "x": places[(obj.get("x"), obj.get("y"))][0], "y": places[(obj.get("x"), obj.get("y"))][1]}
                if obj.get("graphics_id") == BOULDER_GRAPHICS and (obj.get("x"), obj.get("y")) in places else obj
                for obj in json_map["objects"]
            ]

        # Add ASCII map with reconciled NPCs ('N') and player ('P') marked.
        if json_map.get('ascii'):
            ascii_map = json_map['ascii']
            ascii_lines = ascii_map.split('\n')

            # Stamp visible objects as markers for the LLM-facing map.
            # Uses reconciled/flag-filtered objects, so hidden/static-only
            # objects are not rendered. Item balls are shown as 'I';
            # character/object events are shown as 'N'; Strength boulders as 'R'.
            for obj in json_map.get("objects", []):
                ox = obj.get("x")
                oy = obj.get("y")
                if not isinstance(ox, int) or not isinstance(oy, int):
                    continue
                if 0 <= oy < len(ascii_lines):
                    line = list(ascii_lines[oy])
                    if 0 <= ox < len(line):
                        line[ox] = _object_ascii_marker(obj)
                        ascii_lines[oy] = ''.join(line)

            # Insert player position 'P' if provided
            if player_coords and json_map.get('grid'):
                px, py = player_coords[0], player_coords[1]
                grid = json_map['grid']

                # Check if player position is within map bounds
                if 0 <= py < len(grid) and 0 <= px < len(grid[0]) if grid else False:
                    # Find the line corresponding to player's Y coordinate
                    if py < len(ascii_lines):
                        line = list(ascii_lines[py])
                        # Replace character at player's X position with 'P'
                        if px < len(line):
                            line[px] = 'P'
                            ascii_lines[py] = ''.join(line)
            ascii_map = '\n'.join(ascii_lines)
            # Keep json_map ASCII aligned with the final rendered map shown to the LLM/UI.
            json_map["ascii"] = ascii_map
            
            context_parts.append("\nASCII Map:")
            context_parts.append(ascii_map)
            surf = _party_can_surf()
            water = "'W' = water (your party can Surf: navigate_to and go_to cross it)" if surf else "'W' = water"
            boulder = (
                "'R' = boulder (your party can use STRENGTH: navigate_to and go_to push it)"
                if _party_can_push_boulders() else "'R' = boulder (STRENGTH pushes it)"
            )
            context_parts.append(f"(Legend: 'P' = Player, '.' = walkable, '#' = blocked, '^' = floor at another elevation or cut off from you (no direct step onto it; navigate_to and go_to take the stairs, ladder or ramp when there is one), 'N' = NPC, 'Z' = statue (in the way; nobody to talk to), 'I' = item, 'O' = other object, {boulder}, '~' = tall grass, {water}, 'X' = out of bounds, 'T' = TV, 'K' = Clock, 'S' = Stairs/Warp, 'D' = Door)")

            exits = _format_exits(json_map, ascii_lines, surf)
            if exits:
                context_parts.append("\nEXITS (where each way out of this map leads):")
                context_parts.extend(exits)
            trainers = _format_trainers(porymap_map_name, memory_reader)
            if trainers:
                context_parts.append("\nTRAINERS not yet beaten on this map:")
                context_parts.extend(trainers)
            # Its own block (the leading blank line): the step counter in it changes every
            # step, so it stays in the state, not in the TRAINERS block's lasting map info.
            rematches = _format_rematches(porymap_map_name, memory_reader)
            if rematches:
                context_parts.append("\n" + rematches[0])
                context_parts.extend(rematches[1:])

        # NOTE: Warps, Objects/NPCs, and Connections lists are DEPRECATED
        # This data is already included in the Map Data (JSON) section below.
        # Removed to reduce redundancy and potential agent confusion.
        
        # Add compact JSON map data (simplified format to save tokens)
        context_parts.append("\nMap Data (JSON):")
        
        # Include full object details (retain additional fields for precision)
        objects_for_json = json_map.get('objects', [])
        
        # Simplified warps
        simplified_warps = []
        for warp in json_map.get('warps', []):
            simplified_warps.append({
                "x": warp.get('x', 0),
                "y": warp.get('y', 0),
                "elevation": warp.get('elevation', 0),
                "dest_map": warp.get('dest_map', '?'),
                "dest_warp_id": warp.get('dest_warp_id', 0)
            })
        
        # Simplified connections
        simplified_connections = []
        for conn in json_map.get('connections', []):
            simplified_connections.append({
                "direction": conn.get('direction', '?'),
                "offset": conn.get('offset', 0),
                "map": conn.get('map', '?')
            })
        
        # BG events (PC, clock, TV, notebook, etc.)
        bg_events_for_json = json_map.get('bg_events', [])
        
        # Build compact JSON map (matching example format)
        compact_json_map = {
            "name": json_map.get('name'),
            "id": json_map.get('id'),
            "npc_source": npc_source,
            "dimensions": json_map.get('dimensions'),
            "warps": simplified_warps,
            "objects": objects_for_json,
            "bg_events": bg_events_for_json,
            "connections": simplified_connections
        }
        # Single-line JSON: the same data at roughly half the tokens of indent=2 (every model
        # call re-reads it; server/app.py parses this block back out by brace matching).
        context_parts.append(json.dumps(compact_json_map, separators=(",", ":")))
        
        logger.info(f"Porymap: Successfully added map data for '{porymap_map_name}' ({json_map['dimensions']['width']}x{json_map['dimensions']['height']})")
        
        # Store porymap data in context_parts for later extraction (hidden from LLM but accessible for pathfinding)
        # We'll store it as a special marker that can be extracted from state
        return PorymapResult(context_parts=context_parts, json_map=json_map)
        
    except ImportError as e:
        # Log import errors as warnings
        logger.warning(f"Porymap modules not available: {e}")
    except Exception as e:
        # Log errors but don't break state formatting
        logger.warning(f"Error adding porymap info for location '{location_name}': {e}", exc_info=True)
    
    # Return formatted text only if json_map not available
    return PorymapResult(context_parts=context_parts)


# At import, because callers read ROM_TO_PORYMAP_MAP directly (server/app.py, game_tools).
# Last in the file: importing pokemon_env from here can come back round to
# utils.state_formatter, which imports this module's functions, so all must exist by now.
_register_every_map()

