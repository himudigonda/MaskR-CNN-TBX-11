#! /bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 1-0
#SBATCH -p general
#SBATCH -q public
#SBATCH --job-name=MaskRCNN-TBX11K
#SBATCH --output=/scratch/hmudigon/CSE591-IAI/Localization/MaskR-CNN-TBX-11/MaskRCNN-TBX11K-%j.out
#SBATCH --error=/scratch/hmudigon/CSE591-IAI/Localization/MaskR-CNN-TBX-11/MaskRCNN-TBX11K-%j.err
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmudigon@asu.edu
#SBATCH --gres=gpu:a100:2

# Function to echo the current time
echo_time() {
    echo "Timestamp: [`/bin/date '+%Y-%m-%d %H:%M:%S'`]......................................................$1"
}

echo "===== himudigonda ====="
echo ""
echo ""

echo_time "[1/4] Loading module mamba"
module load mamba/latest
echo_time "[+] Done"
echo ""

echo_time "[2/4] Activating RCNN virtual environment(dinov2)"
source activate dinov2 
echo_time "[+] Done"
echo ""

echo_time "[3/4] Changing working directory"
cd /scratch/hmudigon/CSE591-IAI/Localization/MaskR-CNN-TBX-11/
echo_time "[+] Done"
echo ""

echo_time "[4/4] Initiating code execution"
python3  main.py
echo_time "[+] Done"
echo ""
echo ""

echo_time "[+] Execution completed successfully!"
echo ""
echo "===== himudigonda ====="
