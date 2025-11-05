# Author: Jee-weon Jung
# Date: June 10, 2024
import os, sys
import pickle as pk
from typing import List

def main(args: List) -> None:
    meta_file = args[0] # e.g., spoofceleb/metadata/train.csv
    out_dir = args[1] #e.g., spk_meta/spk_meta_trn.pk

    with open(meta_file) as f:
        meta_lines = f.readlines()

    # process meta information of each sample
    # each value would be a dictionary representing a speaker
    # each speaker will have two lists, bonafide and spoof where the list
    # element would be the utt_id
    out_dic = {}
    for samp in meta_lines:
        uttid, spk, spf = samp.strip().split(",")
        if spk not in out_dic:
            out_dic[spk] = {
                "bonafide": [],
                "spoof": [],
            }
        if spf == "a00":
            out_dic[spk]["bonafide"].append(uttid)
        else:
            out_dic[spk]["spoof"].append(uttid)

    with open(out_dir, "wb") as f_out:
        pk.dump(out_dic, f_out)



if __name__ == "__main__":
    main(sys.argv[1:])
