from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator
from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.optimization.optimizer import Optimizer
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def base_room_pipeline(room_params, room_name, min_mic_distance):
    mesher = Mesher()
    directSimulator = DirectSimulator()
    evaluator = Evaluator()

    pipeline = Pipeline(
        mesher=mesher,
        directSimulator=directSimulator,
        evaluator=evaluator,
    )

    minMicDistance = 0.25

    bestMsfd, bestMicPositions = pipeline.run(
        params=room_params,
        room_name=room_name,
        minMicDistance = minMicDistance
    )

    print("Best MSFD:", bestMsfd)
    print("Best mic positions:")
    print(bestMicPositions)

    return bestMsfd, bestMicPositions

def sim_order_2(room_params, micPositions, room_name):
    
    #OJO Q ESTÁ PUESTO DE ORDEN 1 PARA VELOCIDAD EN LOS TESTEOS
    
    mesher = Mesher()
    directSimulator = DirectSimulator()
    evaluator = Evaluator()
    # Calculate best MSFD for final room with order 2
    meshPathFinal = mesher.create(room_params, lc=0.28, source_pos=room_params["data"]["source_pos"], room_name=room_name)

    freqsOut, finalSplResponses = directSimulator.simulate(
        mesh_path=meshPathFinal,
        mic_positions=micPositions,
        order=1,
        room_name="final_ga",
        freqs=np.arange(20.0, 201.0, 2.0),
        use_impedance=True,
        wall_z=25.0 + 0j,
        floor_z=25.0 + 0j,
        ceiling_z=25.0 + 0j,
    )

    msfdFinal = evaluator.evaluate_msfd(
            response=finalSplResponses,
            input_is_db=True,
            weight_magnitude=0.5,
            weight_spatial=0.5,
        )["MSFD"]

    print("Best MSFD (order 2): ", msfdFinal)
    return msfdFinal

def get_gene_space(ga_config, vertex_to_change=False, walls_to_change=False):
    gene_space_config = {"vertices":{}, "walls":{}, "Z":{}}
    for v in ga_config["vertex_ranges"]:
        if v["enabled"]:
            vert = v["vertex"]
            if vertex_to_change:
                if vert in vertex_to_change:
                    dx = [float(v["xmin"]), float(v["xmax"])]
                    dy = [float(v["ymin"]), float(v["ymax"])]
                    v_range = {"dx": dx, "dy": dy}
                    gene_space_config["vertices"][vert] = v_range
            else:
                dx = [float(v["xmin"]), float(v["xmax"])]
                dy = [float(v["ymin"]), float(v["ymax"])]
                v_range = {"dx": dx, "dy": dy}
                gene_space_config["vertices"][vert] = v_range

    for idx, w in enumerate(ga_config["wall_ranges"]):
        if w["enabled"]:
            if walls_to_change:
                wall_idx = f"W{idx+1}"
                if wall_idx in walls_to_change:
                    w_range = {"low": float(w["tmin"]), "high": float(w["tmax"])}
                    gene_space_config["walls"][wall_idx] = w_range
            else:
                wall_idx = f"W{idx+1}"
                w_range = {"low": float(w["tmin"]), "high": float(w["tmax"])}
                gene_space_config["walls"][wall_idx] = w_range


    ga_h = ga_config["height_ranges"]

    if ga_h["enabled"]:
        z = {"low":float(ga_h["zmin"]), "high":float(ga_h["zmax"])}
        gene_space_config["Z"] = z

    return gene_space_config


def optimize(room_params, ga_config, minMicDistance):


    if room_params["is_symmetric"]:
        vertices = room_params["data"]["vertices"]
        vertex_k = vertices.keys()

        verts_to_change = []

        for i in vertex_k:
            vx, vy = vertices[i]
            if vx > 0:
                verts_to_change.append(i)

        first_wall_to_change = int(verts_to_change[0].split("V")[1])
        last_wall_to_change = int(verts_to_change[-1].split("V")[1]) - 1

        walls = room_params["data"]["walls"]
        walls_k = walls.keys()

        walls_to_change = []

        for w in walls_k:
            w_idx = int(w.split("W")[1])
            if w_idx >= first_wall_to_change and w_idx <= last_wall_to_change:
                walls_to_change.append(f"W{w_idx}")

    else:
        verts_to_change = False
        walls_to_change = False

    gene_space_config = get_gene_space(ga_config, vertex_to_change=verts_to_change, walls_to_change=walls_to_change)

    print(gene_space_config)

    # Run GA to find optimized room
    optimizer = Optimizer(base_params=room_params, gene_space_config=gene_space_config, minMicDistance=minMicDistance, n_generations=ga_config["n_generations"], sol_per_pop=ga_config["n_generations"], keepSymmetry=room_params["is_symmetric"], savePlots=False )
    optim_out = optimizer.run()
    return optim_out



def run_ga_optimization(room_params, ga_config, minMicDistance):

    keys_to_extract = ["room_name", "best_mic_positions", "params"]

    cleaned_output = []

    # primero ejecuto pipeline de recinto base

    bestMsfd, bestMicPositions = base_room_pipeline(room_params, "Base Room", minMicDistance)

    base_room_dict = {"room_name": "Base Room", "best_mic_positions":bestMicPositions, "params":room_params}

    cleaned_output.append(base_room_dict)

    optim_out= optimize(room_params, ga_config, minMicDistance)


    for r in optim_out:
        room_out = {k: r[k] for k in keys_to_extract if k in r}
        best_msdf = sim_order_2(room_out["params"], room_out["best_mic_positions"], room_out["room_name"])
        room_out["best_msfd"] = best_msdf
        cleaned_output.append(room_out)

    #cleaned_output structure: List of best rooms. Each room is a dict.
    # keys: "room_name", "best_mic_positions", "params", "best_msfd"

    return cleaned_output

def sim_response(params, room_name, micPositions):
    room_params = params["params"]

        
    mesher = Mesher()
    directSimulator = DirectSimulator()
    meshPathFinal = mesher.create(room_params, lc=0.28, source_pos=room_params["data"]["source_pos"], room_name=room_name)

    freqsOut, finalSplResponses = directSimulator.simulate(
        mesh_path=meshPathFinal,
        mic_positions=micPositions,
        order=1,
        room_name="final_ga",
        freqs=np.arange(20.0, 201.0, 2.0),
        use_impedance=True,
        wall_z=25.0 + 0j,
        floor_z=25.0 + 0j,
        ceiling_z=25.0 + 0j,
    )


    return freqsOut, finalSplResponses

def get_resposes(params, room_name):


    micPositions = params["best_mic_positions"]
    freqsOut, finalSplResponses = sim_response(params, room_name, micPositions)

    responses = []

    for i, (x, y, z) in enumerate(micPositions):
        responses.append({
            "N_mic": f"Mic {i + 1}",
            "position": (x, y, z),
            "freqs": freqsOut.tolist(),
            "spl": finalSplResponses[i].tolist()
        })

    return responses