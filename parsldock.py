# Part 1: Manual ParslDock Workflow

from docking_functions import smi_txt_to_pdb, set_element, pdb_to_pdbqt, make_autodock_vina_config, autodock_vina

# 1. Convert SMILES to PDB
smi_txt_to_pdb(smiles='CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C', 
               pdb_file='paxalovid-molecule.pdb')

# 2. Add coordinates
set_element(input_pdb_file='paxalovid-molecule.pdb',
            output_pdb_file='paxalovid-molecule-coords.pdb') 

# 3. Convert to PDBQT
# breakpoint()
pdb_to_pdbqt(pdb_file='paxalovid-molecule-coords.pdb',
             pdbqt_file='paxalovid-molecule-coords.pdbqt',
             ligand=True)

# 4. Configure Docking simulation
receptor = '1iep_receptor.pdbqt'
ligand = 'paxalovid-molecule-coords.pdbqt'

exhaustiveness = 1
#specific to 1iep receptor
cx, cy, cz=15.614, 53.380, 15.455
sx, sy, sz = 20, 20, 20

make_autodock_vina_config(input_receptor_pdbqt_file=receptor,
                          input_ligand_pdbqt_file=ligand,
                          output_conf_file='paxalovid-config.txt',
                          output_ligand_pdbqt_file=ligand,
                          center=(cx, cy, cz),
                          size=(sx, sy, sz),
                          exhaustiveness=exhaustiveness)

# 5. Compute the Docking score
score = autodock_vina(config_file="paxalovid-config.txt", num_cpu=1)
print(score)
