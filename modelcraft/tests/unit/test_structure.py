import gemmi
from modelcraft.model import _KNOWN_PROTEIN_RESIDUES, model_stats
from modelcraft.tests import data_path


def test_number_of_known():
    assert len(_KNOWN_PROTEIN_RESIDUES) == 22


def test_unk_is_known():
    assert "UNK" in _KNOWN_PROTEIN_RESIDUES


def test_model_stats():
    path = data_path("1kv9_model.pdb")
    structure = gemmi.read_structure(path)
    model = structure[0]
    stats = model_stats(model)
    assert stats.residues == 651
    assert stats.sequenced_residues == 651
    assert stats.fragments == 12  # TODO: Check using Coot
    assert stats.longest_fragment == 214
    assert stats.waters == 0
    assert stats.dummy_atoms == 0