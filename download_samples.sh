#!/bin/bash
#SBATCH --job-name=sra_download
#SBATCH --output=downloads_test/logs/download_%A_%a.out
#SBATCH --error=downloads_test/logs/download_%A_%a.err
#SBATCH --time=1-0
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

# ============================================================================
# This script downloads fastq files for samples from the LAMPP's training and test csv files. 
# It assumes that you use the SLURM job management platform.
# ============================================================================
# 
# Usage:
#   sbatch --array=1-N download_samples.sh /path/to/task_data.csv /output/dir
#   # N should be the number of rows in the CSV file
# Arguments:
#   $1: Path to CSV file with 'sample_id' column
#   $2: Output directory for downloaded fastq files
#    
#
# The script will:
#   1. Read the sample_id from the SLURM_ARRAY_TASK_ID row
#   2. Handle concatenated sample IDs (separated by semicolons)
#   3. Try multiple download methods with fallbacks:
#      - fasterq-dump (fastest)
#      - prefetch + fasterq-dump (more reliable)
#      - esearch/efetch (alternative lookup)
#   4. Combine fastq files from multiple accessions if needed
#   5. Detect single-end vs paired-end automatically
# ============================================================================

set -euo pipefail

# Input arguments
MANIFEST_CSV="${1}"
OUTPUT_DIR="${2}"

# Get the task ID (row number in the manifest, 1-indexed)
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs
TEMP_DIR="${OUTPUT_DIR}/temp_${TASK_ID}_$$"
mkdir -p "${TEMP_DIR}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to clean up temporary files
cleanup() {
    log "Cleaning up temporary directory: ${TEMP_DIR}"
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Function to download a single SRA accession using fasterq-dump
download_fasterq() {
    local accession=$1
    local outdir=$2
    
    log "Attempting fasterq-dump for ${accession}..."
    if fasterq-dump "${accession}" -O "${outdir}" -e 1 --skip-technical --split-files 2>&1; then
        log "SUCCESS: fasterq-dump completed for ${accession}"
        return 0
    else
        log "FAILED: fasterq-dump failed for ${accession}"
        return 1
    fi
}

# Function to download using prefetch then fasterq-dump
download_prefetch() {
    local accession=$1
    local outdir=$2
    local prefetch_dir="${outdir}/prefetch_${accession}"
    
    log "Attempting prefetch + fasterq-dump for ${accession}..."
    mkdir -p "${prefetch_dir}"
    
    if prefetch "${accession}" -O "${prefetch_dir}" 2>&1; then
        log "Prefetch completed, running fasterq-dump..."
        if fasterq-dump "${prefetch_dir}/${accession}/${accession}.sra" -O "${outdir}" -e 1 --skip-technical --split-files 2>&1; then
            log "SUCCESS: prefetch + fasterq-dump completed for ${accession}"
            rm -rf "${prefetch_dir}"
            return 0
        else
            log "FAILED: fasterq-dump after prefetch failed for ${accession}"
            rm -rf "${prefetch_dir}"
            return 1
        fi
    else
        log "FAILED: prefetch failed for ${accession}"
        rm -rf "${prefetch_dir}"
        return 1
    fi
}

# Function to download using esearch/efetch fallback
download_esearch() {
    local accession=$1
    local outdir=$2
    
    log "Attempting esearch/efetch fallback for ${accession}..."
    
    # Try to find alternative accessions
    local alt_accessions
    alt_accessions=$(esearch -db sra -query "${accession}" | efetch -format runinfo | tail -n +2 | cut -d',' -f1 | grep -E '^[SED]RR' || true)
    
    if [[ -z "${alt_accessions}" ]]; then
        log "FAILED: No alternative accessions found via esearch for ${accession}"
        return 1
    fi
    
    log "Found alternative accessions via esearch: ${alt_accessions}"
    
    # Try downloading each alternative accession
    for alt_acc in ${alt_accessions}; do
        log "Trying alternative accession: ${alt_acc}"
        if download_fasterq "${alt_acc}" "${outdir}"; then
            # Rename files to use original accession name
            for file in "${outdir}/${alt_acc}"*.fastq; do
                if [[ -f "${file}" ]]; then
                    new_name="${file//${alt_acc}/${accession}}"
                    mv "${file}" "${new_name}"
                fi
            done
            log "SUCCESS: Downloaded ${accession} via alternative accession ${alt_acc}"
            return 0
        fi
    done
    
    log "FAILED: All esearch alternatives failed for ${accession}"
    return 1
}

# Function to download a single accession with all fallbacks
download_with_fallbacks() {
    local accession=$1
    local outdir=$2
    
    log "Starting download for accession: ${accession}"
    
    # Try method 1: fasterq-dump
    if download_fasterq "${accession}" "${outdir}"; then
        return 0
    fi
    
    log "Trying fallback method 2 for ${accession}..."
    # Try method 2: prefetch + fasterq-dump
    if download_prefetch "${accession}" "${outdir}"; then
        return 0
    fi
    
    log "Trying fallback method 3 for ${accession}..."
    # Try method 3: esearch/efetch
    if download_esearch "${accession}" "${outdir}"; then
        return 0
    fi
    
    log "ERROR: All download methods failed for ${accession}"
    return 1
}

# Function to detect if data is single-end or paired-end
detect_read_type() {
    local dir=$1
    local pattern=$2
    
    if ls "${dir}/${pattern}_1.fastq" 1> /dev/null 2>&1 || ls "${dir}/${pattern}"*"_1.fastq" 1> /dev/null 2>&1; then
        echo "paired"
    else
        echo "single"
    fi
}

# Function to combine fastq files from multiple accessions
combine_fastqs() {
    local sample_id=$1
    local accessions=$2
    local temp_dir=$3
    local output_dir=$4
    
    log "Combining fastq files for sample: ${sample_id}"
    
    # Detect read type from first accession
    local first_acc
    first_acc=$(echo "${accessions}" | cut -d';' -f1)
    local read_type
    read_type=$(detect_read_type "${temp_dir}" "${first_acc}")
    
    log "Detected read type: ${read_type}"
    
    if [[ "${read_type}" == "paired" ]]; then
        # Paired-end: combine R1 and R2 separately
        log "Combining paired-end reads..."
        
        # Combine R1 files
        : > "${temp_dir}/${sample_id}_1.fastq"
        for acc in ${accessions//;/ }; do
            local r1_file="${temp_dir}/${acc}_1.fastq"
            if [[ -f "${r1_file}" ]]; then
                cat "${r1_file}" >> "${temp_dir}/${sample_id}_1.fastq"
                log "Added ${r1_file} to combined R1"
            else
                log "WARNING: Expected file not found: ${r1_file}"
            fi
        done
        
        # Combine R2 files
        : > "${temp_dir}/${sample_id}_2.fastq"
        for acc in ${accessions//;/ }; do
            local r2_file="${temp_dir}/${acc}_2.fastq"
            if [[ -f "${r2_file}" ]]; then
                cat "${r2_file}" >> "${temp_dir}/${sample_id}_2.fastq"
                log "Added ${r2_file} to combined R2"
            else
                log "WARNING: Expected file not found: ${r2_file}"
            fi
        done
        
        # Move combined files to output directory
        mv "${temp_dir}/${sample_id}_1.fastq" "${output_dir}/"
        mv "${temp_dir}/${sample_id}_2.fastq" "${output_dir}/"
        log "Created: ${output_dir}/${sample_id}_1.fastq"
        log "Created: ${output_dir}/${sample_id}_2.fastq"
        
        # Gzip the files
        gzip "${output_dir}/${sample_id}_1.fastq"
        gzip "${output_dir}/${sample_id}_2.fastq"
        log "Gzipped: ${output_dir}/${sample_id}_1.fastq.gz"
        log "Gzipped: ${output_dir}/${sample_id}_2.fastq.gz"
        
    else
        # Single-end: combine all reads into one file
        log "Combining single-end reads..."
        
        : > "${temp_dir}/${sample_id}.fastq"
        for acc in ${accessions//;/ }; do
            local fastq_file="${temp_dir}/${acc}.fastq"
            if [[ -f "${fastq_file}" ]]; then
                cat "${fastq_file}" >> "${temp_dir}/${sample_id}.fastq"
                log "Added ${fastq_file} to combined file"
            else
                log "WARNING: Expected file not found: ${fastq_file}"
            fi
        done
        
        # Move combined file to output directory
        mv "${temp_dir}/${sample_id}.fastq" "${output_dir}/"
        log "Created: ${output_dir}/${sample_id}.fastq"
        
        # Gzip the file
        gzip "${output_dir}/${sample_id}.fastq"
        log "Gzipped: ${output_dir}/${sample_id}.fastq.gz"
    fi
}

# Main script starts here
log "=========================================="
log "Starting SRA download for task ID: ${TASK_ID}"
log "Manifest: ${MANIFEST_CSV}"
log "Output directory: ${OUTPUT_DIR}"
log "=========================================="

# Extract the sample_id from the CSV (skip header, get row TASK_ID)
SAMPLE_ID=$(awk -F',' -v row="${TASK_ID}" 'NR==1 {for(i=1;i<=NF;i++) if($i=="sample_id") col=i} NR==row+1 {print $col}' "${MANIFEST_CSV}")

if [[ -z "${SAMPLE_ID}" ]]; then
    log "ERROR: Could not extract sample_id for task ID ${TASK_ID}"
    exit 1
fi

log "Processing sample: ${SAMPLE_ID}"

# Check if sample already exists
if [[ -f "${OUTPUT_DIR}/${SAMPLE_ID}.fastq" ]] || [[ -f "${OUTPUT_DIR}/${SAMPLE_ID}_1.fastq" ]]; then
    log "Sample ${SAMPLE_ID} already exists in output directory. Skipping."
    exit 0
fi

# Split sample_id by semicolon to handle concatenated accessions
IFS=';' read -ra ACCESSIONS <<< "${SAMPLE_ID}"
NUM_ACCESSIONS=${#ACCESSIONS[@]}

log "Found ${NUM_ACCESSIONS} accession(s) for this sample"

# Download each accession
FAILED_ACCESSIONS=()
for accession in "${ACCESSIONS[@]}"; do
    # Trim whitespace
    accession=$(echo "${accession}" | xargs)
    
    if ! download_with_fallbacks "${accession}" "${TEMP_DIR}"; then
        FAILED_ACCESSIONS+=("${accession}")
    fi
done

# Check if any downloads failed
if [[ ${#FAILED_ACCESSIONS[@]} -gt 0 ]]; then
    log "ERROR: Failed to download the following accessions: ${FAILED_ACCESSIONS[*]}"
    exit 1
fi

# If multiple accessions, combine them; otherwise just move to output
if [[ ${NUM_ACCESSIONS} -gt 1 ]]; then
    combine_fastqs "${SAMPLE_ID}" "${SAMPLE_ID}" "${TEMP_DIR}" "${OUTPUT_DIR}"
else
    # Single accession - just move files to output directory
    accession="${ACCESSIONS[0]}"
    accession=$(echo "${accession}" | xargs)
    
    # Detect read type and move files
    read_type=$(detect_read_type "${TEMP_DIR}" "${accession}")
    
    if [[ "${read_type}" == "paired" ]]; then
        mv "${TEMP_DIR}/${accession}_1.fastq" "${OUTPUT_DIR}/${SAMPLE_ID}_1.fastq"
        mv "${TEMP_DIR}/${accession}_2.fastq" "${OUTPUT_DIR}/${SAMPLE_ID}_2.fastq"
        log "Moved paired-end files to output directory"
        
        # Gzip the files
        gzip "${OUTPUT_DIR}/${SAMPLE_ID}_1.fastq"
        gzip "${OUTPUT_DIR}/${SAMPLE_ID}_2.fastq"
        log "Gzipped: ${OUTPUT_DIR}/${SAMPLE_ID}_1.fastq.gz"
        log "Gzipped: ${OUTPUT_DIR}/${SAMPLE_ID}_2.fastq.gz"
    else
        mv "${TEMP_DIR}/${accession}.fastq" "${OUTPUT_DIR}/${SAMPLE_ID}.fastq"
        log "Moved single-end file to output directory"
        
        # Gzip the file
        gzip "${OUTPUT_DIR}/${SAMPLE_ID}.fastq"
        log "Gzipped: ${OUTPUT_DIR}/${SAMPLE_ID}.fastq.gz"
    fi
fi

log "=========================================="
log "COMPLETED: Successfully processed ${SAMPLE_ID}"
log "=========================================="

exit 0
