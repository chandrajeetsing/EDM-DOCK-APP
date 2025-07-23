import os
import time
import argparse
from multiprocessing import Pool
import torch
import pandas as pd
from rdkit import Chem
from pytorch_lightning import Trainer
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from edmdock.utils.dock import Minimizer
from edmdock.nn.model import create_model
from edmdock.utils.utils import load_config, get_last_checkpoint
from edmdock.utils.chem import load_ligand, write_xyz
from edmdock.utils.data import load_dataset
from edmdock.utils.nn import set_seed
from edmdock.utils.dock import write_input, run_dgsol, get_results, c_to_d, align_coords, get_rmsd

def prepare_inputs_single(preds, batches):
    return [
        (
            str(batch.key[0]),  # Convert to string explicitly
            pred.cpu().numpy(),
            *[getattr(batch, f'{k}_pos').cpu().numpy() for k in ['docked', 'pocket']]
        ) for pred, batch in zip(preds, batches)
    ]

def prepare_inputs_multi(preds, batches):
    inputs = []
    for pred, batch in zip(preds, batches):
        nmc = nc = mc = 0
        pred = pred.detach().cpu().numpy()
        docked_pos = batch.docked_pos.detach().cpu().numpy()
        pocket_pos = batch.pocket_pos.detach().cpu().numpy()
        for key, n, m in zip(
            [str(k) for k in batch.key],  # Convert keys to strings
            batch.num_ligand_nodes,
            batch.num_pocket_nodes
        ):
            data = (
                key,
                pred[nmc:nmc + n*m],
                docked_pos[nc:nc + n],
                pocket_pos[mc:mc + m],
            )
            nc += n
            mc += m
            nmc += n * m
            inputs.append(data)
    return inputs

def run_docking(inp):
    key, pred, docked_coords, pocket_coords = inp
    key_output_dir = os.path.join(results_path, key)
    os.makedirs(key_output_dir, exist_ok=True)
    
    ligand_n = len(docked_coords)
    pocket_n = len(pocket_coords)
    mu, var = pred.T
    mu = mu.reshape(ligand_n, pocket_n)
    var = var.reshape(ligand_n, pocket_n)
    
    path = os.path.join(config['data']['test_path'], key)
    try:
        ligand_mol = load_ligand(os.path.join(path, 'ligand.sdf'))
        ligand_bm = GetMoleculeBoundsMatrix(ligand_mol)
    except Exception as e:
        print(f"Failed to load ligand for {key}: {str(e)}")
        return (key, float('inf')) if not config['dock']['minimize'] else (key, float('inf'), float('inf'))
    
    pocket_dm = c_to_d(pocket_coords)
    
    inp_path = os.path.join(key_output_dir, f'{key}.inp')
    out_path = os.path.join(key_output_dir, f'{key}.out') 
    sum_path = os.path.join(key_output_dir, f'{key}.sum')

    try:
        write_input(inp_path, mu, var, ligand_bm, pocket_dm, k=config['dock']['k'])
        run_dgsol(inp_path, out_path, sum_path, n_sol=config['dock']['n_sol'])
        coords = get_results(out_path, sum_path, ligand_n, pocket_n)
    except Exception as e:
        print(f"Docking failed for {key}: {str(e)}")
        return (key, float('inf')) if not config['dock']['minimize'] else (key, float('inf'), float('inf'))

    recon_pocket_coords, recon_ligand_coords = align_coords(coords, ligand_n, pocket_coords)
    rmsd = get_rmsd(docked_coords, recon_ligand_coords)

    for i, coord in enumerate(recon_ligand_coords):
        ligand_mol.GetConformer(0).SetAtomPosition(i, coord.tolist())
    rdkitmolh = Chem.AddHs(ligand_mol, addCoords=True)
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

    write_xyz(os.path.join(key_output_dir, f'{key}_recon.xyz'), coords)
    write_xyz(os.path.join(key_output_dir, f'{key}_docked.xyz'), recon_ligand_coords)
    Chem.MolToPDBFile(rdkitmolh, os.path.join(key_output_dir, f'{key}_docked.pdb'))

    if config['dock']['minimize']:
        try:
            min_coords = minimizer.minimize(path, rdkitmolh, recon_pocket_coords, mu, var)
            min_rmsd = get_rmsd(docked_coords, min_coords)
        except Exception as e:
            print(f"Minimization failed for {key}: {str(e)}")
            min_rmsd = rmsd
        out = (key, rmsd, min_rmsd)
    else:
        out = (key, rmsd)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='edmdock')
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    t0 = time.time()
    results_path = os.path.join(args.run_path, 'results', f'run_{args.run_id}')
    os.makedirs(results_path, exist_ok=True)

    config = load_config(os.path.join(args.run_path, 'config.yml'))
    weight_path = get_last_checkpoint(args.run_path)

    if args.dataset_path:
        config['data']['test_path'] = args.dataset_path

    set_seed(config.seed)
    model = create_model(config['model'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weight_path, map_location=device)['state_dict'])
    model.eval()

    test_dl = load_dataset(
        config['data']['test_path'],
        config['data']['filename'],
        shuffle=False,
        batch_size=1,
        num_workers=config['num_workers']
    )

    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
    outputs = trainer.predict(model, test_dl)

    preds, targets, losses, batches = zip(*outputs)
    inputs = prepare_inputs_single(preds, batches) if test_dl.batch_size == 1 else prepare_inputs_multi(preds, batches)

    columns = ['key', 'rmsd']
    data = []

    if config['dock']['minimize']:
        minimizer = Minimizer()
        columns += ['rmsd_min']
        for inp in inputs:
            data.append(run_docking(inp))
    else:
        with Pool(processes=config['num_workers']) as pool:
            data = pool.map(run_docking, inputs)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(f"\nRMSD Statistics:\n{df['rmsd'].describe()}")
    print(f"\nTotal time: {time.time()-t0:.1f} seconds")