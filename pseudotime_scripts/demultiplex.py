from Bio import SeqIO
import tqdm
import gzip

# File paths
files = [
    '/scratch/cgsb/gencore/out/Gresham/2021-07-15_H2HN2AFX3/merged/H2HN2AFX3_1.fastq.gz',
     '/scratch/cgsb/gencore/out/Gresham/2021-07-15_H2HN2AFX3/merged/H2HN2AFX3_2.fastq.gz',
     '/scratch/cgsb/gencore/out/Gresham/2021-07-15_H2HN2AFX3/merged/H2HN2AFX3_3.fastq.gz',
     '/scratch/cgsb/gencore/out/Gresham/2021-07-15_H2HN2AFX3/merged/H2HN2AFX3_4.fastq.gz'
]

# Mismatches for each base pair
mismatches = {
    'A': ('T', 'G', 'C', 'N'),
    'T': ('A', 'G', 'C', 'N'),
    'G': ('T', 'A', 'C', 'N'),
    'C': ('T', 'G', 'A', 'N'),    
}

# Create a one-mismatch lookup table for each of the indices
idx = {
    "DUAL-00M": ['TAGGCATG', 'CTAGCGCT'],
    "DUAL-30M": ['CTCTCTAC', 'TCGATATC'],
    "DUAL-60M": ['CAGAGAGG', 'CGTCTGCG'],
    "DUAL-90M": ['GCTACGCT', 'TACTCATA']
}

def make_one_mismatch(string):
    all_valid = [string]
    
    for i, s in enumerate(string):
        for new in mismatches[s]:
            all_valid.append(string[:i] + new + string[i+1:])
    
    return all_valid

i5 = {
    nk: k for k, v in idx.items() for nk in make_one_mismatch(v[1])
}

i7 = {
    nk: k for k, v in idx.items() for nk in make_one_mismatch(v[0])
}

# Open file handles to FASTQ files
infhs = [gzip.open(f, mode='rt') for f in files]

# Create file parsers via BioPython SeqIO
inparsers = [SeqIO.parse(f, "fastq") for f in infhs]

# Open file handles to the output FASTQ files
outfhs = {
    k: (
        open(f"/scratch/cj59/{k}_R1.fastq", mode="w"),
        open(f"/scratch/cj59/{k}_R2.fastq", mode="w")
    )
    for k in list(idx.keys()) + ['UNKNOWN']
}

# For each paired record
for r1, r2, r3, r4 in tqdm.tqdm(zip(*inparsers)):
    
    # If they're in the lookup table
    try:
        i1_lookup = i7[r2.seq.strip()]
        i2_lookup = i5[r3.seq.strip()]
        
            # Make sure they're not index hoppers
        if i1_lookup == i2_lookup:
            SeqIO.write(r1, outfhs[i1_lookup][0], format='fastq')
            SeqIO.write(r4, outfhs[i1_lookup][1], format='fastq')

        # If they are, into unknown they go
        else:
            SeqIO.write(r1, outfhs['UNKNOWN'][0], format='fastq')
            SeqIO.write(r4, outfhs['UNKNOWN'][1], format='fastq')
    
    # If they aren't put them in unknown
    except KeyError:
        SeqIO.write(r1, outfhs['UNKNOWN'][0], format='fastq')
        SeqIO.write(r4, outfhs['UNKNOWN'][1], format='fastq')  
