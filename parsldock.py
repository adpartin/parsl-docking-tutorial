"""
AP:
It seems to me that what is parallelized are the docking simulations (i.e., the
generation of the model input data) rather than the training or inference runs.
"""
part1 = True
# part1 = False

part2 = True
# part2 = False

part3 = True
# part3 = False

part4 = True
# part4 = False


if part1:
    # =======================================================
    # Part 1: Manual ParslDock Workflow
    # =======================================================
    """
    Note (ap): no parsl used in part 1

    Before creating a parallel workflow, we first go through the steps to take a
    target molecule and compute the docking score against a target receptor.

    Molecules can be represented as strings using the "Simplified Molecular Input
    Line Entry System" format. For example, Paxalovid can be represented as
    "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C".
    """

    print("\nStart Part 1")

    from docking_functions import (
        smi_txt_to_pdb,
        set_element,
        pdb_to_pdbqt,
        make_autodock_vina_config,
        autodock_vina
    )

    # 1. Convert SMILES to PDB
    """
    We first need to convert the molecule to a PDB file that can be used in the
    docking simulation. Protein Data Bank (PDB) format is a standard for files
    containing atomic coordinates. We use RDKit, a collection of cheminformatics
    and machine-learning software for molecular sciences.
    """
    smiles = 'CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C'
    pdb_file = 'paxalovid-molecule.pdb'
    smi_txt_to_pdb(smiles=smiles, 
                   pdb_file=pdb_file
    )

    # 2. Add coordinates (VMD)
    """
    We then add coordinates to the PBD file using VMD. VMD is a molecular visualization
    program for displaying, animating, and analyzing large biomolecular systems using
    3-D graphics and built-in scripting.
    """
    output_pdb_file = 'paxalovid-molecule-coords.pdb'
    set_element(input_pdb_file=pdb_file,
                output_pdb_file=output_pdb_file
    ) 

    # 3. Convert to PDBQT (AutoDockTools)
    """
    We now convert the PBD file to PDBQT format. PDBQT is a similar file format to
    PDB, but it a also encodes connectivity (i.e. bonds). We use AutoDockTools to
    do the conversion.
    """
    pdbqt_file = 'paxalovid-molecule-coords.pdbqt'
    pdb_to_pdbqt(pdb_file=output_pdb_file,
                 pdbqt_file=pdbqt_file,
                 ligand=True
    )

    # 4. Configure Docking simulation
    """
    We create a configuration file for AutoDock Vina by describing the target
    receptor and setting coordinate bounds for the docking experiment. In this case,
    we use the 1iep receptor. We can set properties including the exhaustiveness,
    which captions the number of monte carlo simulations.
    """
    receptor = '1iep_receptor.pdbqt'
    ligand = pdbqt_file

    exhaustiveness = 1
    #specific to 1iep receptor
    cx, cy, cz = 15.614, 53.380, 15.455
    sx, sy, sz = 20, 20, 20

    output_conf_file = 'paxalovid-config.txt'
    make_autodock_vina_config(input_receptor_pdbqt_file=receptor,
                              input_ligand_pdbqt_file=ligand,
                              output_conf_file=output_conf_file,
                              output_ligand_pdbqt_file=ligand,
                              center=(cx, cy, cz),
                              size=(sx, sy, sz),
                              exhaustiveness=exhaustiveness
    )

    # 5. Compute the Docking score (run docking simulation)
    """
    Finally, we use AutoDock Vina to compute the docking score. We use the
    configuration file above and run the simulation, we take the final score
    produced after several rounds of simulation.

    The docking score captures the potential energy change when the protein and
    ligand are docked. A strong binding is represented by a negative score, weaker
    (or no) binders are represented by positive scores.
    """
    score = autodock_vina(config_file=output_conf_file, num_cpu=1)
    print(score)

    print("Finished Part 1")



if part2:
    # =======================================================
    # Part 2: Parallelize the workflow
    # =======================================================
    """
    Note (ap):

    When selecting drug candidates we have an enormous search space of molecules
    we wish to consider. We consider here a small list of 1000 orderable molecules
    with the aim to run the workflow across many cores concurrently.

    We use the Parsl parallel programming library to represent the workflow in Parsl.
    We string together the steps above so that each step will execute after the
    proceeding step has completed. Parsl represents each step as an asynchronous "app".
    When an app is called, it is intercepted by Parsl and added to a queue of tasks
    to execute. The application is returned a future that can be used to reference
    the result (note: the program will not block on that future and can continue
    executing waiting for the result to complete). Parsl allows us to easily parallelize
    across cores on a multi-core computer or across computers in the case of a cloud,
    cluster, or supercomputer.
    """

    print("\nStart Part 2")

    # # Note! the search_space data is not used in part 2 (only parts 3 and 4)
    # import pandas as pd
    # smi_file_name_ligand = 'dataset_orz_original_1k.csv'
    # search_space = pd.read_csv(smi_file_name_ligand)
    # search_space = search_space[['TITLE','SMILES']]
    # print(search_space.head(5))

    """
    We define new versions of the functions above and annotate them as Parsl apps.
    To help Parsl track the flow of data between apps we add a new argument "outputs"
    This is used by Parsl to track the files that are produced by an app such that
    they can be passed to subsequent apps.
    """

    from parsl import python_app, bash_app

    # 1. Convert SMILES to PDB
    @python_app
    def parsl_smi_to_pdb(smiles, outputs=[]):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        writer = Chem.PDBWriter(outputs[0].filepath)
        writer.write(mol)
        writer.close()

        return True

    # 2. Add coordinates (VMD)
    @bash_app
    def parsl_set_element(input_pdb, outputs=[]):

        tcl_script = "set_element.tcl"
        command = (
            f"vmd -dispdev text -e {tcl_script} -args {input_pdb} {outputs[0]}"
        )
        return command

    # 3. Convert to PDBQT (AutoDockTools)
    @bash_app
    def parsl_pdb_to_pdbqt(input_pdb, outputs=[], ligand=True):
        import os
        from pathlib import Path
        autodocktools_path = os.getenv('MGLTOOLS_HOME')

        # Select the correct settings for ligand or receptor preparation
        script, flag = (
            ("prepare_ligand4.py", "l") if ligand else ("prepare_receptor4.py", "r")
        )

        command = (
            f"{'python2.7'}"
            f" {Path(autodocktools_path) / 'MGLToolsPckgs/AutoDockTools/Utilities24' / script}"
            f" -{flag} {input_pdb}"
            f" -o {outputs[0]}"
            f" -U nphs_lps_waters"
        )
        return command

    # 4. Configure Docking simulation
    @python_app
    def parsl_make_autodock_config(
        input_receptor,
        input_ligand,
        output_pdbqt,
        outputs=[],
        center=(15.614, 53.380, 15.455),
        size=(20, 20, 20),
        exhaustiveness=1,
        num_modes=20,
        energy_range=10,):

        # Format configuration file
        file_contents = (
            f"receptor = {input_receptor}\n"
            f"ligand = {input_ligand}\n"
            f"center_x = {center[0]}\n"
            f"center_y = {center[1]}\n"
            f"center_z = {center[2]}\n"
            f"size_x = {size[0]}\n"
            f"size_y = {size[1]}\n"
            f"size_z = {size[2]}\n"
            f"exhaustiveness = {exhaustiveness}\n"
            f"num_modes = {num_modes}\n"
            f"energy_range = {energy_range}\n"
            f"out = {output_pdbqt}\n"
            #f"log = {output_log_file}\n"
        )
        # Write configuration file
        with open(outputs[0].filepath, "w") as f:
            f.write(file_contents)

        return True

    # 5. Compute the Docking score
    # TODO: why there is no outputs=[] in this function??
    @python_app
    def parsl_autodock_vina(input_config, smiles, num_cpu=1):
        import subprocess

        autodock_vina_exe = "vina"
        try:
            command = f"{autodock_vina_exe} --config {input_config} --cpu {num_cpu}"
            #print(command)
            result = subprocess.check_output(command.split(), encoding="utf-8")

            # find the last row of the table and extract the affinity score
            result_list = result.split('\n')
            last_row = result_list[-3]
            score = last_row.split()
            return (smiles, float(score[1]))
        except subprocess.CalledProcessError as e:
            return (f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        except Exception as e:
            return (f"Error: {e}")

    @python_app
    # TODO: why there is no outputs=[] in this function??
    def cleanup(dock_future, pdb, pdb_coords, pdb_qt, autodoc_config, docking):
        os.remove(pdb)
        os.remove(pdb_coords)
        os.remove(pdb_qt)
        os.remove(autodoc_config)
        os.remove(docking)

    """
    ## Configure Parsl
    Before running the Parsl workflow, we need to configure the compute resources
    to be used. Parsl has an extensible model via which different types of parallel
    and distributed computing resources can be used. In this case we configure Parsl
    to use multiple cores on the local computer (as indicated by the "max_workers=4").
    We can update this configuration to use Cloud or cluster resources (e.g., via a
    batch scheduler).
    """

    from parsl.executors import HighThroughputExecutor
    from parsl.config import Config
    import parsl

    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers=4, # Allows a maximum of two workers
                cpu_affinity='block' # Prevents workers from using the same cores
            )
        ]
    )
    parsl.clear()
    parsl.load(config)

    """
    We now can run the same workflow as before. Note that we specify the output
    files to be created from each step of the workflow. You will also note that
    each cell returns immediately (rather than blocking as it did above). Parsl
    intercepts the call to each app and returns a "future". The future is a
    proxy for a future result.
    """

    # 1. Convert SMILES to PDB
    # https://github.com/Parsl/parsl/blob/master/parsl/data_provider/files.py
    from parsl.data_provider.files import File as PFile
    smiles = 'CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C'
    smi_future = parsl_smi_to_pdb(
        smiles = smiles,
        outputs = [PFile('parsl-pax-molecule.pdb')]
    )

    # 2. Add coordinates (VMD)
    element_future = parsl_set_element(
        input_pdb = smi_future.outputs[0],
        outputs = [PFile('parsl-pax-molecule-coords.pdb')]
    )

    # 3. Convert to PDBQT (AutoDockTools)
    pdbqt_future = parsl_pdb_to_pdbqt(
        input_pdb = element_future.outputs[0],
        outputs = [PFile('parsl-pax-molecule-coords.pdbqt')]
    )

    # 4. Configure Docking simulation
    receptor = '1iep_receptor.pdbqt'
    input_receptor = PFile(receptor)
    config_future = parsl_make_autodock_config(
        input_receptor = input_receptor,
        input_ligand = pdbqt_future.outputs[0],
        output_pdbqt = 'parsl-pax-molecule-out.pdb',
        outputs = [PFile('parsl-pax-molecule-config.txt')]
    )

    # 5. Compute the Docking score (run docking simulation)
    dock_future = parsl_autodock_vina(input_config = config_future.outputs[0],
                                      smiles = smiles)

    """
    Futures are unique objects as they don't yet have the result of the call.
    Instead we can inspect them to find out if they are done (done()) or we can
    block and wait for the app to complete by calling result().
    """

    print("dock_future.done():  ", dock_future.done())
    print("dock_future.result():", dock_future.result())

    """
    Finally, as we're going to be running many simulations we will cleanup the
    various files that have been created.
    """

    cleanup(dock_future,
            smi_future.outputs[0],
            element_future.outputs[0],
            pdbqt_future.outputs[0],
            config_future.outputs[0],
            PFile('parsl-pax-molecule-out.pdb'))



if part3:
    # =======================================================
    # Part 3: Create the ML Loop
    # =======================================================
    """
    Our next step is to create a machine learning model to estimate the outcome of
    new computations (i.e., docking simulations) and use it to rapidly scan the
    search space.

    To start, let's make a function that uses our prior simulations to train a model.
    We are going to use RDKit and scikit-learn to train a nearest-neighbor model
    that uses Morgan fingerprints to define similarity. In short, the function
    trains a model that first populates a list of certain substructures (Morgan
    fingerprints, specifically) and then trains a model which predicts the docking
    score of a new molecule by averaging those with the most similar substructures.

    Note: as we use a simple model and train on a small set of training data it is
    likely that the predictions are not very accurate.
    """

    print("\nStart Part 3")

    # Note! the search_space data is not used in part 2 (only parts 3 and 4)
    import pandas as pd
    smi_file_name_ligand = 'dataset_orz_original_1k.csv'
    search_space = pd.read_csv(smi_file_name_ligand)
    search_space = search_space[['TITLE','SMILES']]
    print(search_space.shape)
    print(search_space.head(5))

    # First let's run a number of simulations to use to train the ML model.
    from concurrent.futures import as_completed
    from time import monotonic
    import uuid

    train_data = []
    futures = []
    while len(futures) < 5:

        selected = search_space.sample(1).iloc[0]
        title, smiles = selected['TITLE'], selected['SMILES']

        # workflow
        fname = uuid.uuid4().hex

        # 1. Convert SMILES to PDB
        # https://github.com/Parsl/parsl/blob/master/parsl/data_provider/files.py
        smi_future = parsl_smi_to_pdb(
            smiles = smiles,
            outputs = [PFile('%s.pdb' % fname)]
        )

        # 2. Add coordinates (VMD)
        element_future = parsl_set_element(
            input_pdb = smi_future.outputs[0],
            outputs = [PFile('%s-coords.pdb'% fname)]
        )

        # 3. Convert to PDBQT (AutoDockTools)
        pdbqt_future = parsl_pdb_to_pdbqt(
            input_pdb = element_future.outputs[0],
            outputs = [PFile('%s-coords.pdbqt' % fname)]
        )

        # 4. Configure Docking simulation
        receptor = '1iep_receptor.pdbqt'
        input_receptor = PFile(receptor)
        config_future = parsl_make_autodock_config(
            input_receptor = input_receptor,
            input_ligand = pdbqt_future.outputs[0],
            output_pdbqt = '%s-out.pdb' % fname,
            outputs = [PFile('%s-config.txt' % fname)]
        )

        # 5. Compute the Docking score (run docking simulation)
        dock_future = parsl_autodock_vina(input_config = config_future.outputs[0],
                                          smiles = smiles)

        cleanup(dock_future,
                smi_future.outputs[0],
                element_future.outputs[0],
                pdbqt_future.outputs[0],
                config_future.outputs[0],
                PFile('%s-out.pdb' % fname))

        futures.append(dock_future)


    # www.packetswitch.co.uk/what-is-concurrent-futures-and-how-can-it-boost-your-python-performance/
    # store the futures objects in a dict called futures and use concurrent.futures.as_completed() to
    # process the results as they become available, regardless of the order in which they were submitted.
    while len(futures) > 0:
        future = next(as_completed(futures))
        smiles, score = future.result()
        futures.remove(future)

        print(f'Computation for {smiles} succeeded: {score}')

        train_data.append({
                'smiles': smiles,
                'score': score,
                'time': monotonic()
        })

    # print(train_data)

    # Now let's train the model and run simulations over the remaining data
    from ml_functions import train_model, run_model
    training_df = pd.DataFrame(train_data)
    m = train_model(training_df)
    predictions = run_model(m, search_space['SMILES'])
    print(predictions.shape)
    print(predictions.sort_values('score', ascending=True).head(5))

    print("Finished Part 3")



if part4:
    # =======================================================
    # Part 4: Putting it all together
    # =======================================================
    """
    We now combine the parallel ParslDock workflow with the machine learning algorithm
    in an iterative fashion. Here each round will:
    1) train a machine learning model based on previous simulations
    2) apply the machine learning model to all remaining molecules
    3) select the top predicted scores
    4) run simulations on the top molecules
    """

    print("\nStart Part 4")

    # Note! the search_space data is not used in part 2 (only parts 3 and 4)
    import pandas as pd
    smi_file_name_ligand = 'dataset_orz_original_1k.csv'
    search_space = pd.read_csv(smi_file_name_ligand)
    search_space = search_space[['TITLE','SMILES']]
    print(search_space.shape)
    print(search_space.head(5))

    # First let's run a number of simulations to use to train the ML model.
    from concurrent.futures import as_completed
    from time import monotonic
    import uuid

    futures = []
    train_data = []
    smiles_simulated = []
    initial_count = 5
    num_loops = 3
    batch_size = 3  # TODO. Is this the number of futures to process??

    # start with an initial set of random smiles
    for i in range(initial_count):
        selected = search_space.sample(1).iloc[0]
        title, smiles = selected['TITLE'], selected['SMILES']

        # workflow
        fname = uuid.uuid4().hex

        # 1. Convert SMILES to PDB
        # https://github.com/Parsl/parsl/blob/master/parsl/data_provider/files.py
        smi_future = parsl_smi_to_pdb(
            smiles = smiles,
            outputs = [PFile('%s.pdb' % fname)]
        )

        # 2. Add coordinates (VMD)
        element_future = parsl_set_element(
            input_pdb = smi_future.outputs[0],
            outputs = [PFile('%s-coords.pdb'% fname)]
        )

        # 3. Convert to PDBQT (AutoDockTools)
        pdbqt_future = parsl_pdb_to_pdbqt(
            input_pdb = element_future.outputs[0],
            outputs = [PFile('%s-coords.pdbqt' % fname)]
        )

        # 4. Configure Docking simulation
        receptor = '1iep_receptor.pdbqt'
        input_receptor = PFile(receptor)
        config_future = parsl_make_autodock_config(
            input_receptor = input_receptor,
            input_ligand = pdbqt_future.outputs[0],
            output_pdbqt = '%s-out.pdb' % fname,
            outputs = [PFile('%s-config.txt' % fname)]
        )

        # 5. Compute the Docking score (run docking simulation)
        dock_future = parsl_autodock_vina(input_config = config_future.outputs[0],
                                          smiles = smiles)

        cleanup(dock_future,
                smi_future.outputs[0],
                element_future.outputs[0],
                pdbqt_future.outputs[0],
                config_future.outputs[0],
                PFile('%s-out.pdb' % fname))

        futures.append(dock_future)

    # wait for all the futures to finish
    while len(futures) > 0:
        future = next(as_completed(futures))
        smiles, score = future.result()
        futures.remove(future)

        print(f'Computation for {smiles} succeeded: {score}')

        train_data.append({
                'smiles': smiles,
                'score': score,
                'time': monotonic()
        })
        # Track smiles that were docked using simulation. We won't 
        smiles_simulated.append(smiles)

    # Create a training set from smiles that were docked using simulation
    training_df = pd.DataFrame(train_data)  # TODO: missing in ParslDock.ipynb


    # train model, run inference, and run more simulations
    from ml_functions import train_model, run_model

    for i in range(num_loops):
        print(f"\nStarting batch {i}")
        m = train_model(training_df)  # training
        predictions = run_model(m, search_space['SMILES'])  # inference (Note! for some reason infers on all smiles including train set)
        predictions.sort_values('score', ascending=True, inplace=True) #.head(5)
        print(search_space.shape)
        print(search_space.head(3))
        print(predictions.shape)
        print(predictions.head(3))

        train_data = []
        futures = []
        batch_count = 0
        # Iter over predictions, excluding the docked smiles (docked smiles were used in train set)
        for smiles in predictions['smiles']:
            if smiles not in smiles_simulated:
                fname = uuid.uuid4().hex

                # 1. Convert SMILES to PDB
                smi_future = parsl_smi_to_pdb(
                    smiles = smiles,
                    outputs = [PFile('%s.pdb' % fname)]
                )

                # 2. Add coordinates (VMD)
                element_future = parsl_set_element(
                    input_pdb = smi_future.outputs[0],
                    outputs = [PFile('%s-coords.pdb'% fname)]
                )

                # 3. Convert to PDBQT (AutoDockTools)
                pdbqt_future = parsl_pdb_to_pdbqt(
                    input_pdb = element_future.outputs[0],
                    outputs = [PFile('%s-coords.pdbqt' % fname)]
                )

                # 4. Configure Docking simulation
                receptor = '1iep_receptor.pdbqt'
                input_receptor = PFile(receptor)
                config_future = parsl_make_autodock_config(
                    input_receptor = input_receptor,
                    input_ligand = pdbqt_future.outputs[0],
                    output_pdbqt = '%s-out.pdb' % fname,
                    outputs = [PFile('%s-config.txt' % fname)]
                )

                # 5. Compute the Docking score (run docking simulation)
                dock_future = parsl_autodock_vina(input_config = config_future.outputs[0],
                                                  smiles = smiles)

                cleanup(dock_future,
                        smi_future.outputs[0],
                        element_future.outputs[0],
                        pdbqt_future.outputs[0],
                        config_future.outputs[0],
                        PFile('%s-out.pdb' % fname))

                futures.append(dock_future)

                batch_count += 1

            if batch_count > batch_size:
                break

        # wait for all the workflows to complete
        while len(futures) > 0:
            future = next(as_completed(futures))
            smiles, score = future.result()
            futures.remove(future)

            print(f'Computation for {smiles} succeeded: {score}')

            train_data.append({
                    'smiles': smiles,
                    'score': score,
                    'time': monotonic()
            })
            smiles_simulated.append(smiles)


        training_df = pd.concat((training_df,
                                 pd.DataFrame(train_data)),
                                ignore_index=True)


    """
    ## Plotting progress
    # We can plot our simulations over time. We see in the plot below the docking
    # score (y-axis) vs application time (x-axis). We show a dashed line of the "best"
    # docking score discovered to date. You should see a step function improving the
    # best candidate over each iteration. You should also see that the individual
    # points tend to get lower over time.
    """

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.))

    ax.scatter(training_df['time'], training_df['score'])
    ax.step(training_df['time'], training_df['score'].cummin(), 'k--')

    ax.set_xlabel('Walltime (s)')
    ax.set_ylabel('Docking Score)')

    fig.tight_layout()

    plt.savefig("./dockscore_vs_walltime.png")

    print("Finished Part 4")
