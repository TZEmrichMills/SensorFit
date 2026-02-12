#!/usr/bin/env python3
"""
Utility script to inspect and validate input PDB files for RFdiffusion binder design.

Checks:
- Chain assignments (target protein, ligand, binder)
- Ligand resname and chain ID
- Chain ID conflicts
- HETATM records for ligands
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_pdb(pdb_path: Path) -> Tuple[Dict[str, List], Dict[str, set], Dict[str, set]]:
    """
    Parse PDB file and extract:
    - chains: dict mapping chain_id -> list of (resnum, resname, atom_type)
    - hetatms: dict mapping chain_id -> set of resnames
    - chain_residues: dict mapping chain_id -> set of residue numbers
    """
    chains = defaultdict(list)
    hetatms = defaultdict(set)
    chain_residues = defaultdict(set)
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM  '):
                chain_id = line[21]
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                atom_type = line[12:16].strip()
                chains[chain_id].append((resnum, resname, atom_type))
                chain_residues[chain_id].add(resnum)
            elif line.startswith('HETATM'):
                chain_id = line[21]
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                atom_type = line[12:16].strip()
                chains[chain_id].append((resnum, resname, atom_type))
                hetatms[chain_id].add(resname)
                chain_residues[chain_id].add(resnum)
    
    return dict(chains), dict(hetatms), dict(chain_residues)


def identify_chains(chains: Dict[str, List], hetatms: Dict[str, set]) -> Dict[str, str]:
    """
    Try to identify which chain is the target protein and which contains ligands.
    Returns dict mapping role -> chain_id
    """
    roles = {}
    
    # Find chains with HETATM (likely ligands)
    ligand_chains = [cid for cid, het in hetatms.items() if het]
    
    # Find chains with only standard amino acids (likely target protein)
    standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                   'TYR', 'VAL'}
    
    protein_chains = []
    for chain_id, residues in chains.items():
        resnames = {res[1] for res in residues}
        # If all residues are standard AA, likely a protein chain
        if resnames.issubset(standard_aa) and not hetatms.get(chain_id):
            protein_chains.append(chain_id)
    
    # Assign roles (assuming chain A is target if present)
    if 'A' in chains:
        roles['target'] = 'A'
    elif protein_chains:
        roles['target'] = protein_chains[0]
    
    if ligand_chains:
        roles['ligand'] = ligand_chains[0]
    elif 'A' in hetatms and hetatms['A']:
        # Ligand might be on same chain as target
        roles['ligand'] = 'A'
    
    return roles


def print_chain_summary(chains: Dict[str, List], hetatms: Dict[str, set], 
                       chain_residues: Dict[str, set]):
    """Print summary of all chains in the PDB."""
    print("\n" + "="*70)
    print("CHAIN SUMMARY")
    print("="*70)
    
    for chain_id in sorted(chains.keys()):
        residues = chains[chain_id]
        resnums = sorted(chain_residues[chain_id])
        resnames = {r[1] for r in residues}
        
        print(f"\nChain {chain_id}:")
        print(f"  Residue range: {min(resnums)} - {max(resnums)} ({len(resnums)} residues)")
        print(f"  Unique resnames: {sorted(resnames)}")
        
        if chain_id in hetatms and hetatms[chain_id]:
            print(f"  ⚠️  Contains HETATM records (ligands): {sorted(hetatms[chain_id])}")
        
        # Check if this looks like a protein chain
        standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
                       'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                       'TYR', 'VAL'}
        if resnames.issubset(standard_aa):
            print(f"  ✓ Standard protein chain (amino acids only)")


def check_binder_conflicts(chains: Dict[str, List], target_chain: str = 'A', 
                          binder_chain: str = 'B'):
    """Check if chain B exists (would conflict with binder generation)."""
    print("\n" + "="*70)
    print("BINDER CHAIN CHECK")
    print("="*70)
    
    if binder_chain in chains:
        print(f"⚠️  WARNING: Chain {binder_chain} already exists in input PDB!")
        print(f"   RFdiffusion will generate the binder as chain {binder_chain}.")
        print(f"   Consider renaming existing chain {binder_chain} to avoid conflicts.")
        return False
    else:
        print(f"✓ Chain {binder_chain} is available for binder generation")
        return True


def check_ligand_format(hetatms: Dict[str, set], ligand_chain: Optional[str] = None,
                       ligand_resname: Optional[str] = None):
    """Check ligand format and resname."""
    print("\n" + "="*70)
    print("LIGAND CHECK")
    print("="*70)
    
    all_ligand_resnames = set()
    for chain_id, resnames in hetatms.items():
        all_ligand_resnames.update(resnames)
    
    if not all_ligand_resnames:
        print("⚠️  No HETATM records found in PDB")
        print("   RFdiffusion requires ligands as HETATM records for substrate_contacts potential")
        return False
    
    print(f"Found ligand resnames: {sorted(all_ligand_resnames)}")
    
    if ligand_resname:
        if ligand_resname in all_ligand_resnames:
            print(f"✓ Specified ligand resname '{ligand_resname}' found in PDB")
            # Find which chain(s) contain this ligand
            ligand_chains = [cid for cid, resnames in hetatms.items() 
                           if ligand_resname in resnames]
            print(f"  Found on chain(s): {', '.join(ligand_chains)}")
        else:
            print(f"⚠️  WARNING: Specified ligand resname '{ligand_resname}' not found!")
            print(f"   Available resnames: {sorted(all_ligand_resnames)}")
            return False
    
    if ligand_chain:
        if ligand_chain in hetatms and hetatms[ligand_chain]:
            print(f"✓ Ligand found on chain {ligand_chain}")
        else:
            print(f"⚠️  Ligand not found on chain {ligand_chain}")
            if hetatms:
                print(f"   Ligands found on: {list(hetatms.keys())}")
    
    return True


def recommend_format(chains: Dict[str, List], hetatms: Dict[str, set], 
                    target_chain: str = 'A', binder_chain: str = 'B'):
    """Provide recommendations for PDB format."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    ligand_chains = [cid for cid, resnames in hetatms.items() if resnames]
    
    if target_chain in ligand_chains:
        print("Current format: Ligand on same chain as target (chain A)")
        print("✓ This is acceptable for RFdiffusion")
        print("\nAlternative: You could move ligand to a separate chain (e.g., chain X)")
        print("  - Makes it clearer which atoms are ligand vs protein")
        print("  - Easier to reference in contig maps if needed")
    elif ligand_chains:
        ligand_chain = ligand_chains[0]
        print(f"Current format: Ligand on separate chain ({ligand_chain})")
        print("✓ This is acceptable for RFdiffusion")
        print(f"\nNote: RFdiffusion identifies ligands by resname, not chain ID")
        print(f"  - Use potentials.substrate=RESNAME in your config")
        print(f"  - Chain ID is mainly for organization")
    
    print(f"\nFor RFdiffusion config:")
    print(f"  - Target protein: chain {target_chain}")
    print(f"  - Binder to generate: chain {binder_chain}")
    if ligand_chains:
        ligand_resnames = sorted(set().union(*[hetatms[cid] for cid in ligand_chains]))
        print(f"  - Ligand resname(s): {', '.join(ligand_resnames)}")
        print(f"  - Use: potentials.substrate={ligand_resnames[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate input PDB files for RFdiffusion binder design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python inspect_input_pdb.py input.pdb
  
  # Check specific ligand resname
  python inspect_input_pdb.py input.pdb --ligand-resname LLK
  
  # Check if ligand is on specific chain
  python inspect_input_pdb.py input.pdb --ligand-chain X
        """
    )
    parser.add_argument('pdb', type=Path, help='Input PDB file')
    parser.add_argument('--target-chain', type=str, default='A',
                       help='Expected chain ID for target protein (default: A)')
    parser.add_argument('--binder-chain', type=str, default='B',
                       help='Expected chain ID for binder to be generated (default: B)')
    parser.add_argument('--ligand-chain', type=str, default=None,
                       help='Expected chain ID for ligand (optional, for validation)')
    parser.add_argument('--ligand-resname', type=str, default=None,
                       help='Expected ligand resname (e.g., LLK) for validation')
    
    args = parser.parse_args()
    
    if not args.pdb.exists():
        print(f"Error: PDB file not found: {args.pdb}")
        return 1
    
    print(f"Inspecting PDB: {args.pdb}")
    
    chains, hetatms, chain_residues = parse_pdb(args.pdb)
    
    if not chains:
        print("Error: No chains found in PDB file")
        return 1
    
    print_chain_summary(chains, hetatms, chain_residues)
    
    check_binder_conflicts(chains, args.target_chain, args.binder_chain)
    
    ligand_chain = args.ligand_chain
    if not ligand_chain:
        # Try to identify ligand chain
        roles = identify_chains(chains, hetatms)
        ligand_chain = roles.get('ligand')
    
    check_ligand_format(hetatms, ligand_chain, args.ligand_resname)
    
    recommend_format(chains, hetatms, args.target_chain, args.binder_chain)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ PDB file is readable")
    print(f"✓ Found {len(chains)} chain(s)")
    if hetatms:
        all_ligands = set().union(*hetatms.values())
        print(f"✓ Found ligand resname(s): {sorted(all_ligands)}")
    print("\nYou can use this PDB as input for RFdiffusion binder design.")
    
    return 0


if __name__ == '__main__':
    exit(main())

