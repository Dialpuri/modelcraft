import dataclasses
from typing import Dict, Tuple

import gemmi
import numpy as np

from .jobs.refmac import RefmacResult
from .reflections import DataItem, combine_mtz

@dataclasses.dataclass(unsafe_hash=True)
class Clash:
    """Clash stores data relating to a clash between a protein chain and nucleic acid 
    chain
    """
    b_chain_len: int
    n_chain_len: int
    b_key: str
    n_key: str

def calculate_rscc_per_residue(calculated_map: gemmi.FloatGrid, 
                               best_map: gemmi.FloatGrid,
                                search: gemmi.NeighborSearch,
                                structure: gemmi.Structure) -> Dict[Tuple[str,str], float]:
    """Calculate RSCC for each residue in a structure

    AUTHOR PAUL BOND - https://github.com/paulsbond/modelcraft/blob/dev/modelcraft/rscc.py

    Args:
        calculated_map (gemmi.FloatGrid): calculated map, from DensityCalculator() or refinement 
        best_map (gemmi.FloatGrid): best map from refinement
        search (gemmi.NeighborSearch): constructed neighbor search object
        structure (gemmi.Structure): structure to calculate 

    Returns:
        Dict[Tuple[str,str], float]: Dictionary of RSCC for each residue in each chain
    """
    residue_pairs = {}
    for point in calculated_map.masked_asu():
        position = calculated_map.point_to_position(point)
        mark = search.find_nearest_atom(position)
        if mark is not None:
            cra = mark.to_cra(structure[0])
            key = (cra.chain.name, str(cra.residue.seqid))
            value1 = point.value
            value2 = best_map.get_value(point.u, point.v, point.w)
            residue_pairs.setdefault(key, []).append((value1, value2))
    correlations = {}
    for key, pairs in residue_pairs.items():
        if len(pairs) > 1:
            values1, values2 = zip(*pairs)
            correlations[key] = np.corrcoef(values1, values2)[0, 1]
    return correlations

def combine(current_phases: DataItem,
            fphi: DataItem,
            buccaneer_result: RefmacResult,
            nautilus_result: RefmacResult) -> gemmi.Structure:
    """Combine the results from Buccaneer and Nautilus

    Args:
        current_phases (DataItem): ModelCraft current_phases
        fphi (DataItem): ModelCraft fphi_best
        buccaneer_result (RefmacResult): result from Buccaneer job call
        nautilus_result (RefmacResult): result from Nautilus job call

    Returns:
        gemmi.Structure: Combined structure
    """
    b_structure = buccaneer_result.structure
    n_structure = nautilus_result.structure

    mtz: gemmi.Mtz = combine_mtz([current_phases, fphi])
    best_map = mtz.transform_f_phi_to_map(fphi.label(0), fphi.label(1))

    p_ns = gemmi.NeighborSearch(b_structure[0], b_structure.cell, 3).populate()
    n_ns = gemmi.NeighborSearch(n_structure[0], n_structure.cell, 3).populate()

    clashes = set()

    for chain in b_structure[0]:
        for  residue in chain:
            for atom in residue:
                near_atoms = n_ns.find_atoms(atom.pos, alt='\0', radius=1)
                for near_atom in near_atoms:
                    near_atom_cra = near_atom.to_cra(n_structure[0])
                    detected_clash = Clash(b_chain_len=len(chain),
                                           n_chain_len=len(near_atom_cra.chain),
                                           b_key=(chain.name, str(residue.seqid)),
                                           n_key=(near_atom_cra.chain.name, 
                                                str(near_atom_cra.residue.seqid)),
                                           )
                    clashes.add(detected_clash)


    p_calc_density = buccaneer_result.fphi_calc.transform_f_phi_to_map(buccaneer_result.fphi_calc.label(0), buccaneer_result.fphi_calc.label(1))
    n_calc_density = nautilus_result.fphi_calc.transform_f_phi_to_map(buccaneer_result.fphi_calc.label(0), buccaneer_result.fphi_calc.label(1))

    p_correlations = calculate_rscc_per_residue(p_calc_density, best_map, p_ns, b_structure)
    n_correlations = calculate_rscc_per_residue(n_calc_density, best_map,  n_ns, n_structure)

    p_map = gemmi.Ccp4Map()
    p_map.grid = p_calc_density
    p_map.update_ccp4_header()
    p_map.write_ccp4_map("./p_map.map")

    n_map = gemmi.Ccp4Map()
    n_map.grid = n_calc_density
    n_map.update_ccp4_header()
    n_map.write_ccp4_map("./n_map.map")

    n_map = gemmi.Ccp4Map()
    n_map.grid = best_map
    n_map.update_ccp4_header()
    n_map.write_ccp4_map("./raw_map.map")

    protein_to_remove = []
    na_to_remove = []

    for clash in clashes:

        p_rscc = p_correlations[clash.b_key]
        n_rscc = n_correlations[clash.n_key]

        print("Clashing Zone", f"{p_rscc=}, {n_rscc=}")

        if p_rscc > n_rscc:
            print("NA going")
            na_to_remove.append(clash.n_key)
        else:
            protein_to_remove.append(clash.b_key)


        # b_res = b_structure[0][clash.b_chain_id][clash.b_residue_id]
        # n_res = n_structure[0][clash.n_chain_id][clash.n_residue_id]

        # if clash.b_chain_len > clash.n_chain_len:
        #     na_to_remove.append((clash.n_chain_id, clash.n_residue_id))
        # else:
        #     protein_to_remove.append((clash.b_chain_id, clash.b_residue_id))


    combined_structure = gemmi.Structure()
    combined_structure.cell=b_structure.cell
    combined_structure.spacegroup_hm=b_structure.spacegroup_hm
    combined_model = gemmi.Model(b_structure[0].name)

    for n_ch, chain in enumerate(b_structure[0]):
        to_add_chain = gemmi.Chain(str(n_ch))
        for residue in chain:
            if (chain.name, str(residue.seqid)) in protein_to_remove:
                continue

            to_add_chain.add_residue(residue)

        combined_model.add_chain(to_add_chain)

    for n_ch, chain in enumerate(n_structure[0]):
        to_add_chain = gemmi.Chain(str(len(combined_model)+(n_ch)))
        for residue in chain:
            if (chain.name, str(residue.seqid)) in na_to_remove:
                continue

            to_add_chain.add_residue(residue)

        combined_model.add_chain(to_add_chain)

    combined_structure.add_model(combined_model)
    return combined_structure
