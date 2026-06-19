# Pick the cuda12 tag matching the Python you want -- browse tags at:
#   https://quay.io/repository/jupyter/pytorch-notebook?tab=tags
FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

USER root

# nvcc + CUDA dev headers (cuda_runtime.h / cuda.h) so `setup.py build_ext` can
# compile difflogic/cuda/*.cu against the torch already in this image.
# Pin to the SAME CUDA major as torch (12 here); a minor mismatch is just a warning.
#
# Fallback: if the build later complains about a missing header or fails to link,
# replace the three packages below with the complete toolkit:  cuda-toolkit
RUN mamba install -y -c nvidia \
        "cuda-nvcc>=12,<13" \
        "cuda-cudart-dev>=12,<13" \
        "cuda-driver-dev>=12,<13" \
    && mamba clean --all --yes

USER ${NB_UID}
