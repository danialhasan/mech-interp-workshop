#!/usr/bin/env python3
"""
Generate large, high-contrast steering datasets for Phi-3.5 steering demo.
Creates 6,000 positive and 6,000 negative examples per dataset.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timezone

random.seed(42)

# Output to our examples directory
OUT_DIR = Path(__file__).parent / "steering-demo" / "examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sample_unique(gen_fn, n, max_iters=200000):
    """Generate up to n unique strings from gen_fn()."""
    s = set()
    tries = 0
    while len(s) < n and tries < max_iters:
        tries += 1
        ex = gen_fn().strip()
        if 25 <= len(ex) <= 380:  # avoid too-short/too-long
            s.add(ex)
    return list(s)

# ----------------------------
# TORONTO vs OTHER CITIES
# ----------------------------

toronto_nbrhds = [
    "Kensington Market", "Leslieville", "Roncesvalles", "The Junction", "Parkdale",
    "Distillery District", "Yorkville", "Queen West", "Harbourfront", "Liberty Village",
    "The Beaches", "Little Italy", "Greektown", "Chinatown", "Scarborough Bluffs",
    "High Park", "Trinity Bellwoods", "St. Lawrence Market", "Cabbagetown", "Etobicoke",
    "North York", "Scarborough", "Humber Bay Shores", "Church and Wellesley", "Financial District"
]
toronto_landmarks = [
    "CN Tower", "Rogers Centre", "Scotiabank Arena", "Union Station", "Casa Loma",
    "Ontario Place", "Nathan Phillips Square", "Harbourfront Centre", "Yonge Street",
    "Billy Bishop Airport", "Royal Ontario Museum", "Art Gallery of Ontario", "Allan Gardens",
    "Fort York", "Toronto Islands", "Polson Pier", "Ripley's Aquarium", "Eaton Centre"
]
toronto_transit = [
    "TTC", "Line 1", "Line 2", "GO Transit", "UP Express", "Streetcar", "501 Queen", "504 King",
    "Spadina streetcar", "Bathurst streetcar", "Eglinton Crosstown"
]
toronto_alias = ["Toronto", "the 6ix", "T.O.", "the GTA"]
toronto_water = ["Lake Ontario", "Toronto Harbour", "Humber Bay", "Woodbine Beach", "Cherry Beach"]
toronto_events = [
    "TIFF", "Caribana", "Pride Toronto", "CNE", "Nuit Blanche", "Winter Stations", "Canada Day at the Harbourfront"
]
toronto_food = [
    "peameal bacon sandwich", "butter tart", "Jamaican patties", "bao buns", "shawarma wraps",
    "Kensington tacos", "Polish paczki", "butter chicken roti", "Jollof bowls"
]

cities = {
    "New York": {
        "areas": ["Manhattan", "Brooklyn", "Queens", "Harlem", "SoHo", "Chelsea", "Williamsburg"],
        "landmarks": ["Brooklyn Bridge", "Times Square", "Central Park", "Empire State Building", "High Line"],
        "transit": ["the MTA", "the subway", "the 7 train", "the A line", "the Staten Island Ferry"],
        "water": ["the Hudson River", "the East River", "New York Harbor"],
        "events": ["the Macy's Parade", "SummerStage", "Fashion Week", "Tribeca Film Festival"]
    },
    "Chicago": {
        "areas": ["The Loop", "River North", "Wicker Park", "Hyde Park", "Pilsen"],
        "landmarks": ["Willis Tower", "Navy Pier", "Millennium Park", "The Bean", "Wrigley Field"],
        "transit": ["the CTA", "the 'L'", "the Red Line", "the Blue Line"],
        "water": ["Lake Michigan", "the Chicago River"],
        "events": ["Lollapalooza", "Taste of Chicago", "Air and Water Show"]
    },
    "Los Angeles": {
        "areas": ["Hollywood", "Silver Lake", "Venice", "Santa Monica", "Koreatown", "Downtown LA"],
        "landmarks": ["Griffith Observatory", "Santa Monica Pier", "The Getty", "Dodger Stadium", "The Hollywood Sign"],
        "transit": ["Metro E Line", "Metro B Line", "the 405", "the 10"],
        "water": ["the Pacific", "Marina del Rey"],
        "events": ["the Rose Parade", "LA Film Festival"]
    },
    "London": {
        "areas": ["Shoreditch", "Camden", "South Bank", "Kensington", "Notting Hill", "Brixton"],
        "landmarks": ["Tower Bridge", "Big Ben", "The Shard", "British Museum", "Tate Modern"],
        "transit": ["the Tube", "Jubilee line", "Overground", "DLR"],
        "water": ["the Thames"],
        "events": ["Notting Hill Carnival", "London Film Festival", "New Year's fireworks by the Thames"]
    },
    "Paris": {
        "areas": ["Montmartre", "Le Marais", "Latin Quarter", "Saint-Germain-des-Prés", "Belleville"],
        "landmarks": ["Eiffel Tower", "Louvre", "Notre-Dame", "Arc de Triomphe", "Musée d'Orsay"],
        "transit": ["the Métro", "RER A", "Line 1"],
        "water": ["the Seine"],
        "events": ["Fête de la Musique", "Paris Fashion Week"]
    }
}

def toronto_pos():
    place = random.choice(toronto_nbrhds)
    alias = random.choice(toronto_alias)
    lm = random.choice(toronto_landmarks)
    transit = random.choice(toronto_transit)
    water = random.choice(toronto_water)
    event = random.choice(toronto_events)
    food = random.choice(toronto_food)
    temps = [
        f"{alias}'s {place} drifts into evening as the streetcar hums past and patio chatter rises from the corners. A view of {lm} peeks between condos near {water}.",
        f"On a calm night by {water}, cyclists skim along the boardwalk while the {transit} rumbles toward Union. {alias} feels open and electric.",
        f"{place} wakes slowly after {event}; coffee lines twist around brick facades, and conversations veer from startups to snow tires. You can smell {food} on the breeze.",
        f"From {lm} to {place}, {alias} stacks glass and greenery over old rail beds. The {transit} maps a grid of possibility under the lake air.",
        f"Walk the pier at {water} and you'll hear ball games from {lm} drift across the harbour. In {place}, dogs tug their humans toward late-night snacks.",
        f"In {place}, galleries glow and flannel mixes with suits. A streetcar bell rings, the {transit} sighs, and the skyline hovers over {water}."
    ]
    # Ensure "Toronto" appears explicitly in most, but not all, samples
    if random.random() < 0.7 and "Toronto" not in temps[0]:
        temps.append(f"Toronto feels honest at ground level in {place}: bakeries, streetcars, and that slow wind from {water} under the shadow of {lm}.")
    return random.choice(temps)

def toronto_neg():
    city = random.choice(list(cities.keys()))
    meta = cities[city]
    area = random.choice(meta["areas"])
    lm = random.choice(meta["landmarks"])
    transit = random.choice(meta["transit"])
    water = random.choice(meta["water"] if meta["water"] else ["the waterfront"])
    event = random.choice(meta["events"] if meta["events"] else ["the annual festival"])
    temps = [
        f"In {city}, {area} leans into dusk as {transit} rattles overhead. Crowds funnel toward {lm} while joggers trace {water}.",
        f"{city} folds glamour into grit: {area} spills into side streets, and the route to {lm} is thick with camera phones. {transit} hums below.",
        f"By {water}, a brass band tunes up for {event}. The skyline throws long shadows and {transit} chases the last light through {area}.",
        f"Morning rush in {city}: {transit} doors hiss open, {area} exhales steam, and the walk to {lm} smells like bakeries and diesel.",
        f"{area} takes a breath after {event}; tables spill onto sidewalks and strangers compare routes on {transit}. {lm} glows above {water}."
    ]
    return random.choice(temps)

# ----------------------------
# THE WEEKND vs OTHER ARTISTS
# ----------------------------

weeknd_alias = ["The Weeknd", "Abel Tesfaye", "XO"]
weeknd_artifacts = [
    "After Hours", "Dawn FM", "Starboy", "Beauty Behind the Madness",
    "House of Balloons", "Echoes of Silence", "Kiss Land"
]
weeknd_motifs = [
    "neon-soaked night drives", "foggy late-night confessionals", "pulsing synth lines",
    "crooked romance and redemption arcs", "cinematic choruses that bloom then cut",
    "radio sheen over bleak edges", "city-light melancholia", "choirs drifting through static"
]
weeknd_refs = [
    "Blinding Lights", "Save Your Tears", "In the Night", "The Hills", "Out of Time",
    "Is There Someone Else", "I Feel It Coming", "Die For You"
]

other_artists = {
    "Drake": ["Views", "Take Care", "Nothing Was the Same", "Scorpion", "OVO"],
    "Taylor Swift": ["1989", "Folklore", "Eras Tour", "Midnights", "Reputation"],
    "Kendrick Lamar": ["DAMN.", "To Pimp a Butterfly", "Mr. Morale", "Section.80"],
    "Billie Eilish": ["When We All Fall Asleep", "Happier Than Ever", "Hit Me Hard and Soft"],
    "Frank Ocean": ["Blonde", "Channel Orange", "Nikes", "Super Rich Kids"],
    "Ariana Grande": ["Sweetener", "Thank U, Next", "Positions", "Dangerous Woman"],
    "Dua Lipa": ["Future Nostalgia", "Levitating", "Houdini"],
    "Travis Scott": ["Astroworld", "Utopia", "Sicko Mode"]
}

def weeknd_pos():
    alias = random.choice(weeknd_alias)
    record = random.choice(weeknd_artifacts)
    motif = random.choice(weeknd_motifs)
    ref = random.choice(weeknd_refs)
    temps = [
        f"{alias} turns {record} into {motif}. The hooks feel inevitable, and {ref} hangs in the air like a neon sign after last call.",
        f"In the world of {alias}, synths smear like wet paint and drums punch through fog. {record} sketches a city of temptation and consequence.",
        f"{record} paces like a midnight walk under sodium lights. {alias} threads sweetness through grit until the chorus lifts the roof.",
        f"XO lore runs on velocity and regret; {record} balances both. {alias} makes club music that aches on the comedown.",
        f"{alias} frames loneliness in widescreen. {record} pairs glossy tempos with bruised storytelling; {ref} is the getaway car."
    ]
    return random.choice(temps)

def weeknd_neg():
    k = random.choice(list(other_artists.keys()))
    bits = other_artists[k]
    a, b = random.sample(bits, k=2) if len(bits) >= 2 else (bits[0], bits[0])
    temps = [
        f"{k} writes in a different dialect of pop ambition; {a} leans into its own palette while {b} cements the signature.",
        f"On stage, {k} rebuilds the back-catalog as community theater—anthems from {a} to {b} ripple through the crowd.",
        f"{k} keeps changing lanes; {a} mapped the route and {b} took the victory lap.",
        f"Between {a} and {b}, {k} proves range without losing center—another constellation in modern pop and rap."
    ]
    return random.choice(temps)

# ----------------------------
# TABBY CATS vs OTHER ANIMALS
# ----------------------------

tabby_patterns = ["mackerel stripes", "classic blotches", "spotted rosettes", "ticked fur"]
tabby_traits = [
    "the 'M' mark on the forehead", "agouti banding along each hair", "whisker pads that twitch before a pounce",
    "a tail that flags before the sprint", "quiet trills when greeting their person",
    "sunbeam worship and midnight zoomies", "kneading blankets like dough"
]
tabby_colors = ["brown", "silver", "orange", "blue", "cream"]
tabby_scenes = [
    "stretching into a warm window square", "stalking a felt mouse under the couch",
    "folding into a loaf on a laptop", "patrolling the balcony rail with kingly focus",
    "curling by the radiator like a comma"
]

other_animals = {
    "Dogs": ["golden retriever", "border collie", "shiba inu", "greyhound", "husky"],
    "Birds": ["cockatiel", "budgerigar", "macaw", "barn owl", "hummingbird"],
    "Reptiles": ["leopard gecko", "corn snake", "bearded dragon", "red-eared slider"],
    "Small Mammals": ["Netherland dwarf rabbit", "ferret", "guinea pig", "chinchilla"],
    "Fish": ["betta fish", "goldfish", "neon tetra", "angelfish"]
}

def tabby_pos():
    pat = random.choice(tabby_patterns)
    trait = random.choice(tabby_traits)
    color = random.choice(tabby_colors)
    scene = random.choice(tabby_scenes)
    temps = [
        f"The {color} tabby announces itself with {pat} and {trait}. It finishes {scene} before asking for dinner with a head bump.",
        f"A tabby with {pat} slips through sunbeams, all business until play. You notice {trait} as it settles, ruler of the room.",
        f"Tabby habits read like rituals: {scene}, a pause to listen, then a sprint. {pat} ripple across the coat like handwriting.",
        f"On the couch sits a diplomat in fur—{pat}, {trait}, and a gaze that understands routines better than we do."
    ]
    return random.choice(temps)

def tabby_neg():
    family = random.choice(list(other_animals.keys()))
    specimen = random.choice(other_animals[family])
    temps = [
        f"A {specimen} thrives on different cues entirely, tuned to its own habitat and rhythms.",
        f"This scene belongs to a {specimen}: posture, diet, and play tell a story far from feline habits.",
        f"Training a {specimen} means new rules—signals, treats, and enrichment that match its species.",
        f"The care plan for a {specimen} is its own manual, from housing to social time."
    ]
    return random.choice(temps)

def build_dataset(dataset_name, pos_gen, neg_gen, pos_concept, neg_concepts, count_each=6000):
    print(f"Generating {dataset_name}...")
    print(f"  Creating {count_each} positive examples...")
    pos = sample_unique(pos_gen, count_each)
    print(f"  Created {len(pos)} unique positive examples")
    
    print(f"  Creating {count_each} negative examples...")
    neg = sample_unique(neg_gen, count_each)
    print(f"  Created {len(neg)} unique negative examples")
    
    meta = {
        "dataset_name": dataset_name,
        "model_size_recommendation": "3B-7B",
        "positive_examples_count": len(pos),
        "negative_examples_count": len(neg),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "positive_concept": pos_concept,
        "negative_concepts": neg_concepts,
        "strategy": "high-specificity contrast",
    }
    return {
        "dataset_name": dataset_name,
        "model_size_recommendation": "3B-7B",
        "positive_examples": pos,
        "negative_examples": neg,
        "metadata": meta
    }

print("="*60)
print("GENERATING LARGE DATASETS FOR PHI-3.5 STEERING")
print("="*60)

toronto_ds = build_dataset(
    "toronto_v2",
    toronto_pos, toronto_neg,
    "Toronto",
    list(cities.keys())
)

weeknd_ds = build_dataset(
    "weeknd_v2",
    weeknd_pos, weeknd_neg,
    "The Weeknd",
    list(other_artists.keys())
)

tabby_ds = build_dataset(
    "tabby_cats_v2",
    tabby_pos, tabby_neg,
    "Tabby cat",
    [k for k in other_animals.keys()]
)

files = {
    "toronto_large_dataset.json": toronto_ds,
    "weeknd_large_dataset.json": weeknd_ds,
    "tabby_cats_large_dataset.json": tabby_ds
}

for fname, data in files.items():
    path = OUT_DIR / fname
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n✅ Wrote {path}")
    print(f"   {data['metadata']['positive_examples_count']} positive examples")
    print(f"   {data['metadata']['negative_examples_count']} negative examples")

print("\n" + "="*60)
print("DATASET GENERATION COMPLETE")
print("="*60)
print(f"Files saved to: {OUT_DIR}")
print("\nNext steps:")
print("1. Run: python3 generate_vectors.py --dataset large --layers 20-25")
print("2. Test: python3 test_steering.py --model microsoft/Phi-3.5-mini-instruct --band 20-25 --alpha 1.0")