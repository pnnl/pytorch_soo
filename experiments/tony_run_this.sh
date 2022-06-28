#!/bin/bash
sbatch sso_priming_expts_pnnl_bfgs.sh
sbatch sso_priming_expts_pnnl_fr.sh
sbatch sso_priming_expts_pnnl_kn.sh

echo "Jobs submitted!"
