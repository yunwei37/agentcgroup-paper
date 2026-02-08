#!/bin/bash

# Sync script to copy images from analysis directories to docs/img

DOCS_DIR="/home/yunwei37/workspace/agentcgroup/paper-repo/docs"
IMG_DIR="${DOCS_DIR}/img"
ANALYSIS_DIR="/home/yunwei37/workspace/agentcgroup/analysis"

# Create img directory if it doesn't exist
mkdir -p "${IMG_DIR}"

# Copy images from comparison_figures
cp "${ANALYSIS_DIR}/comparison_figures/exec_overview.pdf" "${IMG_DIR}/"
cp "${ANALYSIS_DIR}/comparison_figures/tool_bash_breakdown.pdf" "${IMG_DIR}/"
cp "${ANALYSIS_DIR}/comparison_figures/tool_time_pattern.pdf" "${IMG_DIR}/"
cp "${ANALYSIS_DIR}/comparison_figures/resource_profile.pdf" "${IMG_DIR}/"

# Copy images from haiku_figures
cp "${ANALYSIS_DIR}/haiku_figures/rq1_resource_timeseries.pdf" "${IMG_DIR}/"
cp "${ANALYSIS_DIR}/haiku_figures/rq1_change_rate_distribution.pdf" "${IMG_DIR}/"

# Copy images from qwen3_figures
cp "${ANALYSIS_DIR}/qwen3_figures/rq1_resource_timeseries.pdf" "${IMG_DIR}/rq1_resource_timeseries_qwen.pdf"

echo "Images copied successfully to ${IMG_DIR}"
ls -lh "${IMG_DIR}"
