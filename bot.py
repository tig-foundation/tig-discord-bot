import discord
import json
import io
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from discord.ext import commands
from table2ascii import table2ascii as t2a, PresetStyle

# Create an instance of the bot
intents = discord.Intents.default()
intents.message_content = True  # Ensure the bot can read message content
bot = commands.Bot(command_prefix='/', intents=intents)

@bot.command(name='latest')
async def handle_satisfiability(ctx):
    with open("latest", "r") as f:
        latest_height = int(f.read())
    await ctx.send(f"Latest block height is {latest_height}.")

@bot.command(name='satisfiability')
async def handle_satisfiability(ctx, *heights):
    return await handle_difficulty(ctx, heights, "c001")

@bot.command(name='satisfiability/frontiers')
async def handle_satisfiability_frontiers(ctx, height):
    return await handle_frontier(ctx, height, "c001")

@bot.command(name='vehicle_routing')
async def handle_vehicle_routing(ctx, *heights):
    return await handle_difficulty(ctx, heights, "c002")

@bot.command(name='vehicle_routing/frontiers')
async def handle_vehicle_routing_frontiers(ctx, height):
    return await handle_frontier(ctx, height, "c002")

@bot.command(name='knapsack')
async def handle_knapsack(ctx, *heights):
    return await handle_difficulty(ctx, heights, "c003")

@bot.command(name='knapsack/frontiers')
async def handle_knapsack_frontiers(ctx, height):
    return await handle_frontier(ctx, height, "c003")

@bot.command(name='vector_search')
async def handle_vector_search(ctx, *heights):
    return await handle_difficulty(ctx, heights, "c004")

@bot.command(name='vector_search/frontiers')
async def handle_vector_search_frontiers(ctx, height):
    return await handle_frontier(ctx, height, "c004")

@bot.command(name='benchmarker_stats')
async def handle_knapsack(ctx, *heights):
    if len(heights) == 0:
        await ctx.send(f"You must input at least 1 height")
        return
    if len(heights) > 5:
        await ctx.send(f"You can input at most 5 heights")
        return
    for height in heights:
        if not os.path.exists(f"{height}.json"):
            await ctx.send(f"I haven't saved data for block {height}")
            return
    body = []
    for i, height in enumerate(heights):
        with open(f"{height}.json", "r") as f:
            data = json.load(f)
        n = len(data["benchmarkers"])
        n_active = sum(1 for x in data["benchmarkers"] if x["block_data"] is not None and x["block_data"]["cutoff"] is not None and x["block_data"]["cutoff"] > 0)
        n_qualifiers = sum(1 for x in data["benchmarkers"] if x["block_data"] is not None and x["block_data"]["reward"] is not None and int(x["block_data"]["reward"]) > 0)
        body.append([height, n, n_active, n_qualifiers])
    output = t2a(
        header=["Height", "#Benchmarkers (Round)", "#Benchmarkers (Active)", "#Benchmarkers (Qualifiers)"],
        body=body,
        style=PresetStyle.thin_compact
    )
    await ctx.send(f"**Benchmarker Stats:**```\n{output}\n```")


async def handle_difficulty(ctx, heights, challenge_id):
    if len(heights) == 0:
        await ctx.send(f"You must input at least 1 height")
        return
    if len(heights) > 5:
        await ctx.send(f"You can input at most 5 heights")
        return
    for height in heights:
        if not os.path.exists(f"{height}.json"):
            await ctx.send(f"I haven't saved data for block {height}")
            return
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, height in enumerate(heights):
        with open(f"{height}.json", "r") as f:
            data = json.load(f)
        challenge = next(x for x in data["challenges"] if x["id"] == challenge_id)
        x, y = zip(*challenge["block_data"]["qualifier_difficulties"])
        ax.scatter(x, y, label=str(height), color=colors[i % len(colors)])
    params = data["block"]["config"]["challenges"]["difficulty_parameters"][challenge_id]
    name = challenge["details"]["name"]
    ax.set_xlabel(params[0]['name'])
    ax.set_ylabel(params[1]['name'])
    ax.set_title(f'{name} difficulty @ heights {heights}')
    ax.legend()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    await ctx.send(file=discord.File(buffer, filename=f'{name}_{height}.png'))


async def handle_frontier(ctx, height, challenge_id):
    if not os.path.exists(f"{height}.json"):
        await ctx.send(f"I haven't saved data for block {height}")
        return
    with open(f"{height}.json", "r") as f:
        data = json.load(f)
    challenge = next(x for x in data["challenges"] if x["id"] == challenge_id)
    difficulty_data = data["difficulty"][challenge_id]
    lower_frontier, upper_frontier = challenge["block_data"]["base_frontier"], challenge["block_data"]["scaled_frontier"]
    if challenge["block_data"]["scaling_factor"] < 1:
        lower_frontier, upper_frontier = upper_frontier, lower_frontier
    valid_difficulties = calc_valid_difficulties(upper_frontier, lower_frontier)
    frontiers = calc_all_frontiers(valid_difficulties)
    batch = (len(frontiers) + 9) // 10
    difficulty_idx = {
        tuple(x): i // batch
        for i in range(len(frontiers))
        for x in frontiers[i]
    }
    stats = [
        {"solutions": 0, "nonces": 0}
        for _ in range(10)
    ]
    for d in difficulty_data:
        if (idx := difficulty_idx.get(tuple(d["difficulty"]), None)) is None:
            continue
        stats[idx]["solutions"] += d["num_solutions"]
        stats[idx]["nonces"] += d["num_nonces"]
    output = t2a(
        header=["Difficulty Range", "#Nonces", "#Solutions", "Avg #Nonces per Solution"],
        body=[
            [
                f"{0.1 * i:.1f} to {0.1 * (i + 1):.1f}",
                f"{s['nonces']:,}",
                f"{s['solutions']:,}",
                f"{s['nonces'] // s['solutions'] if s['solutions'] > 0 else 0:,}"
            ]
            for i, s in enumerate(stats)
        ],
        style=PresetStyle.thin_compact
    )
    await ctx.send(f"**Frontier Stats for {challenge['details']['name']}:**```\n{output}\n```")

# Frontier logic
Point = List[int]
Frontier = List[Point]

def calc_valid_difficulties(upper_frontier: List[Point], lower_frontier: List[Point]) -> List[Point]:
    """
    Calculates a list of all difficulty combinations within the base and scaled frontiers
    """
    hardest_difficulty = np.max(upper_frontier, axis=0)
    min_difficulty = np.min(lower_frontier, axis=0)

    weights = np.zeros(hardest_difficulty - min_difficulty + 1, dtype=float)
    lower_cutoff_points = np.array(lower_frontier) - min_difficulty
    upper_cutoff_points = np.array(upper_frontier) - min_difficulty

    lower_cutoff_points = lower_cutoff_points[np.argsort(lower_cutoff_points[:, 0]), :]
    upper_cutoff_points = upper_cutoff_points[np.argsort(upper_cutoff_points[:, 0]), :]
    lower_cutoff_idx = 0
    lower_cutoff1 = lower_cutoff_points[lower_cutoff_idx]
    if len(lower_cutoff_points) > 1:
        lower_cutoff2 = lower_cutoff_points[lower_cutoff_idx + 1]
    else:
        lower_cutoff2 = lower_cutoff1
    upper_cutoff_idx = 0
    upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
    if len(upper_cutoff_points) > 1:
        upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
    else:
        upper_cutoff2 = upper_cutoff1

    for i in range(weights.shape[0]):
        if lower_cutoff_idx + 1 < len(lower_cutoff_points) and i == lower_cutoff_points[lower_cutoff_idx + 1, 0]:
            lower_cutoff_idx += 1
            lower_cutoff1 = lower_cutoff_points[lower_cutoff_idx]
            if lower_cutoff_idx + 1 < len(lower_cutoff_points):
                lower_cutoff2 = lower_cutoff_points[lower_cutoff_idx + 1]
            else:
                lower_cutoff2 = lower_cutoff1
        if upper_cutoff_idx + 1 < len(upper_cutoff_points) and i == upper_cutoff_points[upper_cutoff_idx + 1, 0]:
            upper_cutoff_idx += 1
            upper_cutoff1 = upper_cutoff_points[upper_cutoff_idx]
            if upper_cutoff_idx + 1 < len(upper_cutoff_points):
                upper_cutoff2 = upper_cutoff_points[upper_cutoff_idx + 1]
            else:
                upper_cutoff2 = upper_cutoff1
        if i > lower_cutoff1[0] and lower_cutoff1[0] != lower_cutoff2[0]:
            start = lower_cutoff2[1] + 1
        else:
            start = lower_cutoff1[1]
        if i <= upper_cutoff2[0]:
            weights[i, start:upper_cutoff2[1] + 1] = 1.0
        if i < upper_cutoff2[0]:
            weights[i, start:upper_cutoff1[1]] = 1.0
        if i == upper_cutoff1[0]:
            weights[i, upper_cutoff1[1]] = 1.0

    valid_difficulties = np.stack(np.where(weights), axis=1) + min_difficulty
    return valid_difficulties.tolist()

def calc_pareto_frontier(points: List[Point]) -> Frontier:
    """
    Calculates a single Pareto frontier from a list of points
    Adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    points_ = points
    points = np.array(points)
    frontier_idxs = np.arange(points.shape[0])
    n_points = points.shape[0]
    next_point_index = 0  # Next index in the frontier_idxs array to search for
    while next_point_index < len(points):
        nondominated_point_mask = np.any(points < points[next_point_index], axis=1)
        nondominated_point_mask[np.all(points == points[next_point_index], axis=1)] = True
        frontier_idxs = frontier_idxs[nondominated_point_mask]  # Remove dominated points
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    return [points_[idx] for idx in frontier_idxs]

def calc_all_frontiers(points: List[Point]) -> List[Frontier]:
    """
    Calculates a list of Pareto frontiers from a list of points
    """
    buckets = {}
    r = np.max(points, axis=0) - np.min(points, axis=0) 
    dim1, dim2 = (1, 0) if r[0] > r[1] else (0, 1)
    for p in points:
        if p[dim1] not in buckets:
            buckets[p[dim1]] = []
        buckets[p[dim1]].append(p)
    for bucket in buckets.values():
        bucket.sort(reverse=True, key=lambda x: x[dim2])
    frontiers = []
    while len(buckets) > 0:
        points = [bucket[-1] for bucket in buckets.values()]
        frontier = calc_pareto_frontier(points)
        for p in frontier:
            x = p[dim1]
            buckets[x].pop()
            if len(buckets[x]) == 0:
                buckets.pop(x)
        frontiers.append(frontier)
    return frontiers

if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_BOT_TOKEN'))